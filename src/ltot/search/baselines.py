from typing import List, Tuple, Callable, Optional
import math, random
from ..scorers.vlm import vlm_score

def cot_prompt(q: str) -> str:
    return f"Think step by step.\nQuestion: {q}\nAnswer:"

def scorer_for(task: str):
    if task in ("gsm_plus","gsm_hard","math_500","game24"):
        return lambda model, task, text: vlm_score(model, "math/logic question", text)
    elif task in ("humaneval","mbpp_lite"):
        def score_code(model, task, text):
            ok = "def " in text; return (1.0 if ok else 0.0, 0)
        return score_code
    else:
        return lambda model, task, text: (0.5, 0)

def tot_baseline(model, task, q: str, budget_tokens: int, beam: int=5, max_depth: int=8,
                 exploration_scorer: Optional[Callable]=None, return_tokens: bool=False,
                 verifier: Optional[Callable]=None, early_stop: bool=False) -> Tuple[str,int,int]:
    score = exploration_scorer or scorer_for(task)
    frontier: List[Tuple[str,float]] = [("", 0.0)]
    tokens_spent = 0
    expansions = 0
    for depth in range(max_depth):
        candidates = []
        prompts = [f"{cot_prompt(q)} {path}" for (path, _) in frontier]
        outs, toks = model.generate(prompts, max_tokens=max(16, budget_tokens//(max_depth*beam)))
        tokens_spent += int(sum(toks)); expansions += len(outs)
        for o in outs:
            v, vcost = score(model, task, o); tokens_spent += vcost
            candidates.append((o.strip(), float(v)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        frontier = candidates[:beam]
        if early_stop and verifier is not None:
            if any(verifier(o[0]) >= 1.0 for o in frontier):
                break
        if tokens_spent >= budget_tokens: break
    return (frontier[0][0], tokens_spent, expansions) if return_tokens else (frontier[0][0], tokens_spent, expansions)

def mcts_pw_baseline(model, task, q: str, budget_tokens: int, rollouts: int=64, c_puct: float=1.0,
                     alpha: float=0.5, c_pw: float=1.5, max_depth: int=8,
                     exploration_scorer: Optional[Callable]=None, return_tokens: bool=False,
                     verifier: Optional[Callable]=None, early_stop: bool=False) -> Tuple[str,int,int]:
    score = exploration_scorer or scorer_for(task)
    class Node:
        __slots__=("text","parent","children","N","W","P")
        def __init__(self, text="", parent=None, P=1.0):
            self.text=text; self.parent=parent; self.children=[]; self.N=0; self.W=0.0; self.P=P
    def ucb(par, ch):
        Q = 0.0 if ch.N==0 else ch.W/ch.N
        U = c_puct*ch.P*math.sqrt(max(1, par.N))/(1+ch.N)
        return Q+U
    root = Node("")
    tokens_spent = 0
    expansions = 0
    for _ in range(rollouts):
        node=root; depth=0
        while True:
            depth+=1
            allow_k = 1 + int(c_pw * (node.N ** alpha))
            if len(node.children) < allow_k:
                out, toks = model.generate([cot_prompt(q) + (" " + node.text if node.text else "")], max_tokens=max(16, budget_tokens//(rollouts*2)), temperature=0.7)
                tokens_spent += int(toks[0]); child = Node(out[0].strip(), parent=node, P=1.0/max(1,allow_k))
                node.children.append(child); node = child; expansions += 1; break
            node = max(node.children, key=lambda c: ucb(node, c))
            if depth >= max_depth: break
        v, vcost = score(model, task, node.text); tokens_spent += vcost
        p=node
        while p is not None:
            p.N += 1; p.W += float(v); p=p.parent
        if early_stop and verifier is not None and verifier(node.text) >= 1.0:
            break
        if tokens_spent >= budget_tokens: break
    if not root.children: return ("", tokens_spent, expansions) if return_tokens else ("", tokens_spent, expansions)
    best = max(root.children, key=lambda c: (0.0 if c.N==0 else c.W/c.N))
    return (best.text, tokens_spent, expansions) if return_tokens else (best.text, tokens_spent, expansions)
