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
        for i, r in enumerate(ds):
            yield {"qid": f"gsmplus-{i}", "question": r["question"], "answer": str(r["answer"])}

    elif task == "gsm_hard":
        base = load_dataset("openai/gsm8k", "main", split="test")
        ids_path = canon.get("gsm_hard_ids")
        if not ids_path or not os.path.exists(ids_path):
            if strict:
                raise FileNotFoundError("Missing canonical GSM-Hard ids; set allow_fallback_heuristics=true.")
            hard = [r for r in base if len(r["question"])>130]
            for i, r in enumerate(hard):
                yield {"qid": f"gsmhard-{i}", "question": r["question"], "answer": r["answer"].split('####')[-1].strip()}
        else:
            keep = [int(x) for x in _read_ids(ids_path)]
            for i in keep:
                r = base[i]
                yield {"qid": f"gsmhard-{i}", "question": r["question"], "answer": r["answer"].split('####')[-1].strip()}

    elif task == "math_500":
        ds = load_dataset("hendrycks/competition_math", split="test")
        ids_path = canon.get("math500_ids")
        if not ids_path or not os.path.exists(ids_path):
            if strict: raise FileNotFoundError("Missing canonical math500_ids.txt")
            idx = list(range(len(ds))); random.shuffle(idx); idx = idx[:500]
        else:
            idx = [int(x) for x in _read_ids(ids_path)]
        for i in idx:
            r = ds[i]
            yield {"qid": f"math500-{i}", "question": r["problem"], "answer": r["solution"]}

    elif task == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        for r in ds:
            yield {"qid": f"humaneval-{r['task_id']}", "prompt": r["prompt"], "tests": r["test"], "entry_point": r["entry_point"]}

    elif task == "mbpp_lite":
        ds = load_dataset("Muennighoff/mbpp", split="test")
        ids_path = canon.get("mbpp_ids")
        if not ids_path or not os.path.exists(ids_path):
            if strict: raise FileNotFoundError("Missing canonical mbpp_lite_ids.txt")
            ids = sorted({min(i, len(ds)-1) for i in range(0, len(ds), 10)})[:100]
        else:
            ids = [int(x) for x in _read_ids(ids_path)]
        for i in ids:
            r = ds[i]
            yield {"qid": f"mbpp-{r['task_id']}", "prompt": r["text"], "code": r.get("code",""), "tests": r.get("test_list", [])}

    elif task == "game24":
        path = canon.get("game24")
        if not path or not os.path.exists(path):
            if strict: raise FileNotFoundError("Missing canonical game24_quads.txt")
            quads = [(1,3,4,6),(2,2,6,6),(3,3,8,8),(5,5,5,1)]
        else:
            quads = [tuple(int(t) for t in ln.split()) for ln in _read_ids(path)]
        for i, p in enumerate(quads):
            yield {"qid": f"g24-{i}", "digits": p}
    else:
        raise ValueError(f"Unknown task {task}")
