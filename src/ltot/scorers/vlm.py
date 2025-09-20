from typing import Tuple
import random, re

# Robust 0..1 float parser (mirrors the consistency scorer’s behavior)
_NUM = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|0?\.\d+)(?!\d)")

_PROMPTS = [
    ("You are a strict plausibility scorer. Rate the plausibility that the "
     "DRAFT REASONING leads to a correct solution to the QUESTION. "
     "Return only a single number in [0,1].\n---\nQUESTION:\n{question}\n"
     "DRAFT REASONING:\n{text}\nScore:"),
    ("Output a single calibrated confidence in [0,1] that the following "
     "step sequence is on track to solve the problem.\n---\nProblem:\n{question}\n"
     "Trace:\n{text}\nConfidence:"),
    ("Judge whether the partial solution below is likely correct if continued, "
     "using general world/math knowledge. Respond with only a number between 0 and 1.\n---\n"
     "Q:\n{question}\nPartial solution:\n{text}\nPlausibility:")
]

def _parse_float_01(s: str) -> float:
    m = _NUM.search(s)
    if not m: return 0.5
    try:
        x = float(m.group(0))
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.5

def vlm_score(model, question: str, text: str, temp_range=(0.0, 0.0)) -> Tuple[float, int]:
    """
    LM-scored plausibility used during exploration when no exact verifier is available.
    Returns (score in [0,1], token_cost).
    """
    t = float(random.uniform(*temp_range)) if temp_range else 0.0
    prompt = random.choice(_PROMPTS).format(question=str(question)[:3000], text=str(text)[:3000])
    outs, toks = model.generate([prompt], max_tokens=8, temperature=t, top_p=0.95)
    return _parse_float_01(outs[0]), int(toks[0])
