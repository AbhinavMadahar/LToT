from typing import Tuple
import re

_NUM = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|0?\.\d+)(?!\d)")

_PROMPTS = [
    "Rate [0..1] whether the NEW line logically follows from the PREVIOUS state. Return only the number.\n---\n[PREVIOUS]\n{prev}\n[NEW]\n{new}\nScore:",
    "Output a single number in [0,1] for step consistency (entailment) of NEW from PREVIOUS.\n---\nPREVIOUS:\n{prev}\nNEW:\n{new}\nScore:",
]

def _parse_float_01(s: str) -> float:
    m = _NUM.search(s)
    if not m:
        return 0.5
    try:
        x = float(m.group(0))
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.5

def c_local_score(model, prev: str, new: str, temperature: float = 0.0) -> Tuple[float, int]:
    """Lightweight LM step-consistency (c_local) in [0,1]. Returns (score, token_cost)."""
    prompt = _PROMPTS[0].format(prev=prev[-2000:], new=new[:2000])
    outs, toks = model.generate([prompt], max_tokens=8, temperature=temperature, top_p=0.95)
    return _parse_float_01(outs[0]), int(toks[0])
