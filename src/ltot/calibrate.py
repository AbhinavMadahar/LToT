import json, os
from statistics import median

def _median(arr): 
    a = list(arr)
    return float(median(a)) if a else 0.0

def calibrate(llm, tasks, budgets, methods, seed, tol_pp=2.0, sample_per_task=24, max_iters=4):
    """
    Bisection-style tuner. Adjusts:
      - CoT: max_new_tokens
      - ToT: beam
      - MCTS-PW: rollouts
      - LToT: initial_lateral_width (N0)
    to match median tokens ~= target budget per (task,method) within ±tol% over a small sample.
    """
    os.makedirs("results", exist_ok=True)
    state = {"seed": int(seed), "knobs": {}}

    def measure_tokens(method, task, item, knob_val, target_budget):
        from .run_support import dry_run_tokens
        return dry_run_tokens(llm, method, task, item, knob_val, target_budget)

    for task, items in tasks.items():
        state["knobs"].setdefault(task, {})
        for method in methods:
            if method=="ToT":
                lo, hi = 1, 16
            elif method=="MCTS-PW":
                lo, hi = 8, 256
            elif method=="LToT":
                lo, hi = 32, 1024
            elif method=="CoT":
                lo, hi = 32, max(64, int(budgets["target"]*2))
            else:
                continue
            target = budgets["target"]
            mid = lo
            for _ in range(max_iters):
                mid = int((lo+hi)//2)
                toks = []
                for i, item in enumerate(items[:sample_per_task]):
                    toks.append(measure_tokens(method, task, item, mid, target))
                med_tokens = _median(toks)
                err = 100.0*(med_tokens - target)/max(1.0, target)
                if abs(err) <= tol_pp: break
                if med_tokens > target: hi = max(lo, mid-1)
                else: lo = min(hi, mid+1)
            state["knobs"][task][method] = int(mid)

    with open("results/calibration.json","w") as f:
        json.dump(state, f, indent=2)
    return state
