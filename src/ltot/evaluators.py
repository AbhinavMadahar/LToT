
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json, re, subprocess, tempfile, textwrap, os, sys

NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")

def normalize_answer(a: str) -> str:
    a = a.strip().lower()
    a = re.sub(r"\s+", " ", a)
    return a

def exact_match_numeric(pred: str, gold: str) -> bool:
    p = NUM_RE.findall(pred)
    g = NUM_RE.findall(gold)
    if g:
        return "".join(g) == "".join(p)
    return normalize_answer(pred) == normalize_answer(gold)

@dataclass
class MathVerifier:
    def score(self, pred: str, gold: str) -> float:
        return 1.0 if exact_match_numeric(pred, gold) else 0.0

@dataclass
class QAGate:
    tau_v: float = 0.85
    tau_c: float = 0.75
    tighten_only_lm: bool = True

    def pass_gate(self, plaus: float, cpath: float, only_lm_checker: bool=False) -> bool:
        tv = self.tau_v
        tc = self.tau_c + (0.1 if (self.tighten_only_lm and only_lm_checker) else 0.0)
        return (plaus >= tv) and (cpath >= tc)
