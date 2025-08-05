#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lateral Tree-of-Thoughts (LToT) – Vanilla ToT runner
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Stdlib & third-party imports
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, logging, os, random, re, sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np                   # YAML-safe numeric ops
import yaml                           # pip install pyyaml
import torch                         # pip install torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Utility functions
# ─────────────────────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Make a filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", text).strip("-").lower()
    return slug[:max_len]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")


def make_run_dir(base: Path, exp_name: str, override: str | None = None) -> Path:
    if base.name == "-":
        return None  # black-hole mode
    base.mkdir(parents=True, exist_ok=True)
    if override:
        run_dir = base / override
        run_dir.mkdir(parents=False, exist_ok=False)
        return run_dir
    uid = uuid4().hex[:8]
    run_dir = base / f"{slugify(exp_name)}--{utc_timestamp()}--{uid}"
    run_dir.mkdir()
    return run_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataclass config loader
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SearchCfg:
    beam_size: int = 3
    max_depth: int = 6
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class ModelCfg:
    name_or_path: str
    dtype: str = "auto"           # "float16", "auto", etc.


@dataclass
class ExperimentCfg:
    tag: str
    seed: int = 42
    dataset_file: str = None      # path to prompts/answers CSV
    model: ModelCfg = None
    search: SearchCfg = SearchCfg()


def load_cfg(path: str | Path) -> ExperimentCfg:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["model"] = ModelCfg(**raw["model"])
    raw["search"] = SearchCfg(**raw.get("search", {}))
    return ExperimentCfg(**raw)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tree-of-Thought (BFS beam search) implementation
# ─────────────────────────────────────────────────────────────────────────────
def tot_search(
    prompt: str,
    tokenizer,
    model,
    cfg: SearchCfg,
    device: torch.device,
) -> Tuple[str, List[dict]]:
    """
    Returns the best answer string and the full trace list.
    Each trace dict contains: step, text, score (value = −len as placeholder).
    """
    beam: List[Tuple[str, float]] = [(prompt, 0.0)]
    trace: List[dict] = [{"step": 0, "text": prompt, "value": 0.0}]
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    for depth in range(1, cfg.max_depth + 1):
        candidates: List[Tuple[str, float]] = []
        for ctx, _ in beam:
            inputs = tokenizer(ctx, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, generation_config=gen_cfg)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            # naive split: keep only continuation
            continuation = decoded[len(ctx) :]
            val = -len(continuation)  # placeholder value function
            candidates.append((ctx + continuation, val))
            trace.append({"step": depth, "text": ctx + continuation, "value": val})
        # beam prune
        candidates.sort(key=lambda t: t[1], reverse=True)
        beam = candidates[: cfg.beam_size]

    best_answer, best_val = beam[0]
    return best_answer.strip(), trace


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # CLI
    parser = argparse.ArgumentParser(description="Lateral Tree-of-Thoughts (Vanilla ToT)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--configuration", required=True, help="Path to config YAML")
    parser.add_argument(
        "--results",
        required=True,
        help='Directory to write results (use "-" to discard)',
    )
    parser.add_argument("--results-subdirectory", default=None,
                        help="Override auto-generated run dir name")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ltot")

    # Load config
    cfg = load_cfg(args.configuration)
    log.info("Experiment tag: %s  (seed=%s)", cfg.tag, cfg.seed)

    # Seed
    set_global_seed(cfg.seed)

    # Prepare run directory
    run_dir = make_run_dir(Path(args.results), cfg.tag, args.results_subdirectory)
    if run_dir:
        (run_dir / "config.yaml").write_text(Path(args.configuration).read_text())

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model %s on %s…", cfg.model.name_or_path, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=cfg.model.dtype if cfg.model.dtype != "auto" else None,
    )
    model.eval()

    # Load dataset (very simple CSV with `id,prompt,answer`)
    dataset = []
    if cfg.dataset_file:
        import csv

        with open(cfg.dataset_file) as fh:
            reader = csv.DictReader(fh)
            dataset = list(reader)
    else:
        log.warning("No dataset_file specified; running single synthetic prompt.")
        dataset = [{"id": "example-0", "prompt": "What is 2+2?", "answer": "4"}]

    metrics_rows = []
    trace_fh = None
    if run_dir:
        trace_fh = (run_dir / "traces.jsonl").open("w")

    # Run loop
    log.info("Running ToT on %d items…", len(dataset))
    correct = 0
    for item in dataset:
        ans, trace = tot_search(
            prompt=item["prompt"],
            tokenizer=tokenizer,
            model=model,
            cfg=cfg.search,
            device=device,
        )
        is_correct = ans.strip() == str(item["answer"]).strip()
        correct += int(is_correct)

        metrics_rows.append(
            {
                "id": item["id"],
                "answer": ans,
                "reference": item["answer"],
                "correct": int(is_correct),
                "tree_depth": max(t["step"] for t in trace),
                "num_nodes": len(trace),
            }
        )
        if trace_fh:
            trace_fh.write(json.dumps({"id": item["id"], "trace": trace}) + "\n")

    if trace_fh:
        trace_fh.close()

    # Write metrics & summary
    accuracy = correct / len(dataset)
    if run_dir:
        import csv

        with (run_dir / "metrics.csv").open("w", newline="") as fh:
            fieldnames = metrics_rows[0].keys()
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

        summary = {
            "tag": cfg.tag,
            "seed": cfg.seed,
            "model": cfg.model.name_or_path,
            "beam_size": cfg.search.beam_size,
            "max_depth": cfg.search.max_depth,
            "accuracy": accuracy,
            "datetime_utc": utc_timestamp(),
            "items": len(dataset),
        }
        (run_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    log.info("Finished. Accuracy = %.3f", accuracy)


# ── script entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
