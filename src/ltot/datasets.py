from datasets import load_dataset
import json, os, random
from typing import Iterable, Dict, List, Tuple

def _read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_task(task: str, seed: int) -> Iterable[Dict]:
    import yaml
    with open("configs/experiments.yaml","r") as yf:
        ycfg = yaml.safe_load(yf)
    canon = ycfg.get("datasets",{}).get("canonical_lists",{})
    strict = not bool(ycfg.get("datasets",{}).get("allow_fallback_heuristics", False))
    random.seed(seed)

    if task == "gsm_plus":
        ds = load_dataset("qintongli/GSM-Plus", split="test")
        ids_path = canon.get("gsm_plus_ids")
        if ids_path and os.path.exists(ids_path):
            keep = [int(x) for x in _read_ids(ids_path)]
            for i in keep:
                r = ds[i]
                yield {"qid": f"gsmplus-{i}", "question": r["question"], "answer": str(r["answer"])}
        else:
            # Fallback: stream entire split (or sample if allow_fallback_heuristics=True)
            sample_n = 250
            idxs = list(range(len(ds)))
            if not strict:
                random.shuffle(idxs); idxs = idxs[:sample_n]
            for i in idxs:
                r = ds[i]
                yield {"qid": f"gsmplus-{i}", "question": r["question"], "answer": str(r["answer"])}
