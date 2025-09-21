#!/usr/bin/env python3
"""
scripts/diag_local.py

Tiny, ARC‑free diagnostic that exercises CoT / ToT / MCTS‑PW / LToT end‑to‑end,
and emits the SAME record shapes your paper uses so you can trial
"forecast → empirical" replacement on a miniature artifact.

Defaults are CPU‑safe (tiny model, small budgets). Override via env vars:

  LTOT_DIAG_MODEL  = sshleifer/tiny-gpt2        # any HF causal LM id
  LTOT_DIAG_BUDGET = 96                         # per‑item token budget
  LTOT_DIAG_SEED   = 1
  LTOT_DIAG_TASKS  = gsm_plus,humaneval         # tasks w/o canonical lists
  LTOT_DIAG_ITEMS  = 2                          # items per task
  LTOT_DTYPE       = float32                    # dtype for LocalLM (CPU‑safe)
"""

from __future__ import annotations
import os, sys, json, random
from pathlib import Path
from typing import List

# --- Make the package import robust whether run with -m or as a script ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # type: ignore

from ltot.inference.backends import LocalLM, hf_model_id
from ltot.datasets import load_task
from ltot.search.baselines import tot_baseline, mcts_pw_baseline
from ltot.search.ltot_controller import LToTController
from ltot.config import LToTParams
from ltot.run import build_prompt, verifier_for, scorer_for_exploration
from ltot.util.artifact_jsonl import ArtifactWriter


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and str(v).strip() != "" else default


def main() -> None:
    # --- Tiny, CPU‑friendly defaults; all overridable via env ---
    model_id  = _env("LTOT_DIAG_MODEL",  "sshleifer/tiny-gpt2")
    budget    = int(_env("LTOT_DIAG_BUDGET", "96"))
    seed      = int(_env("LTOT_DIAG_SEED",   "1"))
    tasks     = [t.strip() for t in _env("LTOT_DIAG_TASKS", "gsm_plus,humaneval").split(",") if t.strip()]
    max_items = int(_env("LTOT_DIAG_ITEMS",  "2"))
    dtype     = _env("LTOT_DTYPE",           "float32")   # robust on CPU

    random.seed(seed)
    Path("results/raw").mkdir(parents=True, exist_ok=True)

    # --- Load paper config (bars/plateau/lambdas/mainline params come from here) ---
    with open("configs/experiments.yaml", "r", encoding="utf-8") as yf:
        ycfg = yaml.safe_load(yf) or {}

    plateau_cfg = ycfg.get("plateau", {}) or {}
    # Bars need kappa/delta from controllers.ltot (paper parameters)
    bars_cfg = {
        **(ycfg.get("bars", {}) or {}),
        "kappa": ycfg.get("controllers", {}).get("ltot", {}).get("kappa", 1.0),
        "delta": ycfg.get("controllers", {}).get("ltot", {}).get("delta", 0.02),
    }
    lambdas = (
        ycfg.get("consistency", {}).get("lambdas", {"logic": 0.7, "syntax": 0.2, "constraints": 0.1})
        or {"logic": 0.7, "syntax": 0.2, "constraints": 0.1}
    )
    mainline_cfg = dict(ycfg.get("controllers", {}).get("ltot", {}) or {})
    # Wire envelope & dual‑gate like the main runner does
    mainline_cfg["envelope_cfg"]  = ycfg.get("envelope", None)
    mainline_cfg["dual_gate_cfg"] = ycfg.get("consistency", {}).get("qa_dual_gate", {"enabled": False})

    # Keep it small so a tiny model can run on CPU in seconds
    n0_default = int(mainline_cfg.get("initial_lateral_width", 128))
    mainline_cfg["initial_lateral_width"] = min(n0_default, 16)

    # --- Build a tiny local model (dtype float32 by default for CPU‑only boxes) ---
    llm = LocalLM(hf_model_id(model_id), dtype=dtype)

    # Single artifact sink for this diagnostic's raw events; aggregator will ingest it
    aw = ArtifactWriter("results/raw/diag_local.jsonl")

    def _log_cb_factory(task: str, qid: str):
        """Return a callback that attaches context to LR‑SC/LToT diagnostics."""
        def _log_cb(rec: dict) -> None:
            payload = dict(rec)
            payload.update({
                "task": task, "qid": qid, "model": model_id,
                "method": "LToT", "budget": budget, "seed": seed
            })
            aw.write(payload)
        return _log_cb

    # Optional noisy‑v settings (disabled by default here)
    noisy_cfg = ycfg.get("noisy_v_study", {"enabled": False, "temp_low": 0.0, "temp_high": 0.0}) or {}

    # --- Iterate tasks (only those that don't require canonical lists by default) ---
    for task in tasks:
        # Collect a few items; skip tasks that are fail‑closed in this repo
        items: List[dict] = []
        try:
            for it in load_task(task, seed):
                items.append(it)
                if len(items) >= max_items:
                    break
        except Exception as e:
            print(f"[diag] SKIP task={task}: {e}")
            continue

        for item in items:
            qid = item.get("qid", "unknown")
            q   = item.get("question") or item.get("prompt") or str(item.get("digits"))
            V   = verifier_for(task, item)

            # Exploration scorer matches paper logic (unit‑test subsets for code; vLM otherwise)
            score_expl = scorer_for_exploration(
                task,
                noisy_cfg if (noisy_cfg.get("enabled", False)) else {"temp_low": 0.0, "temp_high": 0.0},
                item
            )

            # ------------------------- CoT -------------------------
            texts, toks = llm.generate([build_prompt(task, item)(q)], max_tokens=max(8, budget // 2))
            out = texts[0]; tok = int(toks[0])
            aw.write({
                "kind": "run", "task": task, "qid": qid, "model": model_id, "method": "CoT",
                "budget": budget, "seed": seed, "pred": out, "score": float(V(out)),
                "tokens": tok, "expansions": 1, "wall_s": 0.0
            })

            # ------------------------- ToT -------------------------
            out, tok, exps = tot_baseline(
                llm, task, q, budget_tokens=budget, beam=3,
                exploration_scorer=score_expl, return_tokens=True, verifier=V, early_stop=True
            )
            aw.write({
                "kind": "run", "task": task, "qid": qid, "model": model_id, "method": "ToT",
                "budget": budget, "seed": seed, "pred": out, "score": float(V(out)),
                "tokens": int(tok), "expansions": int(exps), "wall_s": 0.0
            })

            # ---------------------- MCTS‑PW -----------------------
            out, tok, exps = mcts_pw_baseline(
                llm, task, q, budget_tokens=budget, rollouts=16,
                exploration_scorer=score_expl, return_tokens=True, verifier=V, early_stop=True
            )
            aw.write({
                "kind": "run", "task": task, "qid": qid, "model": model_id, "method": "MCTS-PW",
                "budget": budget, "seed": seed, "pred": out, "score": float(V(out)),
                "tokens": int(tok), "expansions": int(exps), "wall_s": 0.0
            })

            # ------------------------- LToT ------------------------
            ctrl = LToTController(
                llm, LToTParams(), V,
                plateau_cfg=plateau_cfg, bars_cfg=bars_cfg,
                lambdas=lambdas, mainline_cfg=mainline_cfg, early_stop=True
            )

            # Pass a callback that writes *every* LR‑SC/LToT diagnostic event as its own JSONL row
            _log_cb = _log_cb_factory(task, qid)

            text, tok, rung_costs_all, _, exps = ctrl.run(
                build_prompt(task, item), q, budget, task,
                scorer=score_expl, rng=random.Random(seed),
                return_logs=True, log_callback=_log_cb
            )

            # Emit rung_costs explicitly so the aggregator can compute rung CV / counts
            aw.write({
                "kind": "rung_costs", "task": task, "qid": qid, "model": model_id, "method": "LToT",
                "budget": budget, "seed": seed,
                "rung_costs": [int(x) for x in (rung_costs_all or [])]
            })

            # LToT run row with N0 so cost‑law fit logic can operate (even on tiny runs)
            aw.write({
                "kind": "run", "task": task, "qid": qid, "model": model_id, "method": "LToT",
                "budget": budget, "seed": seed, "pred": text, "score": float(V(text)),
                "tokens": int(tok), "expansions": int(exps), "wall_s": 0.0,
                "N0": int(mainline_cfg.get("initial_lateral_width", 16))
            })

    aw.close()
    print("[diag] wrote results/raw/diag_local.jsonl")
    print("[diag] You can now aggregate to build results/ltot_artifact.jsonl and figures/main_equal_compute.svg:")
    print("       python -m ltot.run aggregate \\")
    print("         --inputs results/raw --inputs_width results/raw_width \\")
    print("         --inputs_ablate results/raw_ablate --inputs_latency results/raw_latency \\")
    print("         --artifact results/ltot_artifact.jsonl --fig figures/main_equal_compute.svg")


if __name__ == "__main__":
    main()
