#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lateral Tree-of-Thoughts (LToT)

This script runs experiments for the Lateral Tree-of-Thoughts reasoning architecture.
It reads an experimental configuration file, runs the experiment accordingly,
and saves the results in a corresponding specified output directory.

Vanilla ToT is also supported, and can be used by setting the lateral width to zero.
"""


from __future__ import annotations

import argparse
import datetime
import huggingface_hub
import logging
import itertools
import json
import numpy as np
import random
import re
import sys
import time
import torch
import yaml

from dataclasses import dataclass
from datasets import Dataset, load_dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple, cast
from uuid import uuid4


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


def slugify(text: str, max_length: int = 40) -> str:
    slug_regex = re.compile(r'[^a-zA-Z0-9_-]+')
    slug = slug_regex.sub('-', text).strip('-').lower()
    if len(slug) > max_length:
        raise ValueError(f'Slug \'{slug}\' exceeds maximum length of {max_length} characters.')
    return slug


def make_results_directory(base: Path,
                           experiment_label: str,
                           override: Optional[str] = None) \
    -> Optional[Path]:
    if base.name == '-':
        return None  # black-hole mode

    base.mkdir(parents=True, exist_ok=True)
    if override:
        results_directory = base / override
        results_directory.mkdir(parents=False, exist_ok=False)
        return results_directory
    uid = uuid4().hex[:8]
    results_directory = base / f'{slugify(experiment_label)}--{timestamp()}--{uid}'
    results_directory.mkdir()

    return results_directory


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SearchConfiguration:
    num_beams: int
    max_depth: int
    max_tokens: int
    temperature: float
    top_p: float


@dataclass
class ModelConfiguration:
    name_or_path: str
    dtype: str = "auto"           # "float16", "auto", etc.

@dataclass
class DatasetConfiguration:
    hub_name: str                # e.g., 'gsm8k'
    config: str                  # e.g., 'main' (GSM-8K has only one config)
    split: str                   # 'train' or 'test'
    text_field: str
    answer_field: str
    items: Optional[int] = None  # number of question-answer pairs to consider;
                                 # if None, all items are used


@dataclass
class ExperimentConfiguration:
    tag: str
    seed: Optional[int]
    dataset: DatasetConfiguration
    model: ModelConfiguration
    search: SearchConfiguration


def load_configuration(configuration_file_path: Path) -> ExperimentConfiguration:
    with open(configuration_file_path) as configuration_file:
        raw = yaml.safe_load(configuration_file)
    raw['model'] = ModelConfiguration(**raw['model'])
    raw['search'] = SearchConfiguration(**raw.get('search', {}))
    raw['dataset'] = DatasetConfiguration(**raw.get('dataset', {}))
    return ExperimentConfiguration(**raw)


def tot_search(
    prompt: str,
    tokenizer,
    model,
    configuration: SearchConfiguration,
    device: torch.device,
) -> Tuple[str, List[Dict]]:
    beam: List[str] = [prompt]                   # contexts kept at this depth
    trace: List[Dict] = [{"step": 0, "text": prompt, "value": 0.0}]

    generation_configuration = GenerationConfig(
        do_sample=True,
        num_beams=configuration.num_beams,       # HF manages beam pruning
        renormalize_logits=True,                 # crucial for beam+sampling
        temperature=configuration.temperature,
        top_p=configuration.top_p,
        max_new_tokens=configuration.max_tokens,
    )

    for depth in range(1, configuration.max_depth + 1):
        next_beam: List[str] = []

        for context in beam:
            inputs = tokenizer(context, return_tensors='pt').to(device)

            # HF returns `num_beams` sequences already ranked best to worst
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    generation_config=generation_configuration,
                    num_return_sequences=configuration.num_beams,
                )

            decoded_sequences = tokenizer.batch_decode(out, skip_special_tokens=True)

            # keep HF’s ranking; simple placeholder value = -len(continuation)
            for sequence in decoded_sequences:
                continuation = sequence[len(context):]
                value = -len(continuation)
                trace.append({'step': depth, 'text': sequence, 'value': value})
                next_beam.append(sequence)

        # keep only the top `num_beams` continuations overall
        # (they are already ordered per context, but we may have >num_beams total)
        next_beam = next_beam[: configuration.num_beams]
        beam = next_beam

    best_answer = beam[0].strip()   # first seq after final pruning
    return best_answer, trace


def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Lateral Tree-of-Thoughts (LToT) Application')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--configuration',
                        type=str,
                        required=True,
                        help='Path to the experimental configuration file')
    parser.add_argument('--results',
                        type=str,
                        required=True,
                        help='Directory to write the results to (use "-" to discard results). '
                             'A subdirectory with the name `<name-of-experiment>--YYYY-MM-DD-HH-MM-SS--UID` '
                             'will be created to store the results of the experimental run. Here, '
                             '`<name-of-experiment>` is the name of the experiment as specified in the '
                             'configuration file, `YYYY-MM-DD-HH-MM-SS` is the timestamp of the run (start time) at UTC+0, '
                             'and `UID` is a unique identifier for the run.')
    parser.add_argument('--results-subdirectory',
                        type=str,
                        default=None,
                        help='For the rare cases where you want to specify the exact subdirectory name '
                             'for the results, you can do so here. Will error if the subdirectory already exists.')
    parser.add_argument('--load-in-4bit',
                        action='store_true',
                        default=False,
                        help='Whether to load the model in 4-bit precision using bitsandbytes. '
                             'This can significantly reduce memory usage and enable running larger models on limited hardware.')
    args = parser.parse_args()

    # Configure logging
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("ltot")

    # Main application logic
    logger.info('Starting experiment...')

    configuration = load_configuration(Path(args.configuration))
    logger.info('Experiment tag: %s  (seed=%s)', configuration.tag, configuration.seed)

    if configuration.seed is not None:
        set_global_seed(configuration.seed)

    results_directory = make_results_directory(
        Path(args.results),
        configuration.tag,
        args.results_subdirectory,
    )
    if results_directory:
        (results_directory / 'configuration.yaml').write_text(Path(args.configuration).read_text())
        logger.info('Results will be saved to: %s', results_directory)
        # Add a file handler to also persist logs to experiment.log
        log_file = results_directory / 'experiment.log'
        if not any(getattr(h, 'baseFilename', None) == str(log_file) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            ))
            logger.addHandler(file_handler)
            logger.info('File logging enabled at %s', log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading model %s on %s...', configuration.model.name_or_path, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(configuration.model.name_or_path, use_fast=True)
    except huggingface_hub.errors.GatedRepoError as error:
        logger.error('Encountered an error when loading the tokenizer.\n'
                     'This is most likely a result of a lack of appropriate authentication with HuggingFace.\n'
                     'Please ensure that you are logged in to your HuggingFace account (directions in README.md), and try again.\n'
                     'Original error message reproduced below.\n%s', error)
        sys.exit(1)

    bits_and_bytes_configuration = BitsAndBytesConfig(load_in_4bit=args.load_in_4bit,
                                                      bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        configuration.model.name_or_path,
        device_map='auto' if device.type == 'cuda' else None,
        torch_dtype=configuration.model.dtype if configuration.model.dtype != 'auto' else None,
        quantization_config=bits_and_bytes_configuration if args.load_in_4bit else None,
    )
    model.eval()

    dataset = cast(Dataset, load_dataset(
        configuration.dataset.hub_name,
        configuration.dataset.config,
        split=configuration.dataset.split,
    ))
    assert isinstance(dataset, Dataset)
    dataset = dataset.map(
        lambda _, idx: {'id': idx},
        with_indices=True,
    )

    metrics_rows = []
    trace_file = None
    if results_directory:
        trace_file = (results_directory / 'traces.jsonl').open('w')

    logger.info('Running experiment')
    correct = 0
    for item in itertools.islice(dataset, configuration.dataset.items):
        answer, trace = tot_search(
            prompt=item[configuration.dataset.text_field],
            tokenizer=tokenizer,
            model=model,
            configuration=configuration.search,
            device=device,
        )
        is_correct = answer.strip() == str(item[configuration.dataset.answer_field]).strip()
        correct += int(is_correct)

        metrics_rows.append(
            {
                'id': item['id'],
                'answer': answer,
                'reference': item['answer'],
                'correct': int(is_correct),
                'tree_depth': max(t['step'] for t in trace),
                'num_nodes': len(trace),
            }
        )

        if trace_file:
            trace_file.write(json.dumps({'id': item['id'], 'trace': trace}) + '\n')

    if trace_file:
        trace_file.close()

    # Write metrics & summary
    accuracy = correct / len(dataset)
    if results_directory:
        import csv

        with (results_directory / "metrics.csv").open("w", newline="") as fh:
            fieldnames = metrics_rows[0].keys()
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

        summary = {
            'tag': configuration.tag,
            'seed': configuration.seed,
            'model': configuration.model.name_or_path,
            'num_beams': configuration.search.num_beams,
            'max_depth': configuration.search.max_depth,
            'accuracy': accuracy,
            'datetime_utc': timestamp(),
            'items': len(dataset),
        }
        (results_directory / 'summary.yaml').write_text(yaml.safe_dump(summary, sort_keys=False))

    logger.info('Finished. Accuracy = %.3f', accuracy)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('Experiment interrupted by user.')

        EXIT_CODE_FOR_KEYBOARD_INTERRUPT = 130
        sys.exit(EXIT_CODE_FOR_KEYBOARD_INTERRUPT)
