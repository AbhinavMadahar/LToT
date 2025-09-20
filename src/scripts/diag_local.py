#!/usr/bin/env python3
import os, json, random, yaml
from pathlib import Path

from ltot.inference.backends import LocalLM, hf_model_id
from ltot.datasets import load_task
from ltot.search.baselines import tot_baseline, mcts_pw_baseline
from ltot.search.ltot_controller import LToTController
from ltot.config import LToTParams
from ltot.run import build_prompt, verifier_for, scorer_for_exploration
from ltot.util.artifact_jsonl import ArtifactWriter

def main():
    model_id = os.environ.get("LTOT_DIAG_MODEL", "sshleifer/tiny-gpt2")
    budget   = int(os.environ.get("LTOT_DIAG_BUDGET", "96"))
    seed     = int(os.environ.get("LTOT_DIAG_SEED", "1"))
    tasks    = [t.strip() for t in os.environ.get("LTOT_DIAG_TASKS","gsm_plus,humaneval").split(",")]
    max_items= int(os.environ.get("LTOT_DIAG_ITEMS","2"))

    random.seed(seed)
    Path("results/raw").mkdir(parents=True, exist_ok=True)

    with open("configs/experiments.yaml","r") as yf:
        ycfg = yaml.safe_load(yf)
    plateau_cfg = ycfg.get("plateau", {})
    bars_cfg    = {**ycfg.get("bars", {}),
                   "kappa": ycfg["controllers"]["ltot"]["kappa"],
                   "delta": ycfg["controllers"]["ltot"]["delta"]}
    lambdas     = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
    mainline_cfg= dict(ycfg["controllers"]["ltot"])
    mainline_cfg["initial_lateral_width"] = min(int(mainline_cfg.get("initial_lateral_width",128)), 16)
    mainline_cfg["admission_gate_cfg"] = ycfg.get("consistency",{}).get("admission_gate", {"enabled": False})

    llm = LocalLM(hf_model_id(model_id))
    aw = ArtifactWriter("results/raw/diag_local.jsonl")

    for task in tasks:
        n = 0
        for item in load_task(task, seed):
            n += 1
            if n > max_items: break
            q = item.get("question") or item.get("prompt") or str(item.get("digits"))
            V = verifier_for(task, item)
            noisy_cfg = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
            score_expl = scorer_for_exploration(task, noisy_cfg if noisy_cfg.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)

            texts, toks = llm.generate([build_prompt(task,item)(q)], max_tokens=max(8, budget//2))
            out = texts[0]; tok = int(toks[0])
            aw.write({"kind":"run","task":task,"qid":item["qid"],"model":model_id,"method":"CoT",
                      "budget":budget,"seed":seed,"pred":out,"score":float(V(out)),
                      "tokens":tok,"expansions":1,"wall_s":0.0})

            out, tok, exps = tot_baseline(llm, task, q, budget_tokens=budget, beam=3,
                                          exploration_scorer=score_expl, return_tokens=True,
                                          verifier=V, early_stop=True)
            aw.write({"kind":"run","task":task,"qid":item["qid"],"model":model_id,"method":"ToT",
                      "budget":budget,"seed":seed,"pred":out,"score":float(V(out)),
                      "tokens":int(tok),"expansions":int(exps),"wall_s":0.0})

            out, tok, exps = mcts_pw_baseline(llm, task, q, budget_tokens=budget, rollouts=16,
                                              exploration_scorer=score_expl, return_tokens=True,
                                              verifier=V, early_stop=True)
            aw.write({"kind":"run","task":task,"qid":item["qid"],"model":model_id,"method":"MCTS-PW",
                      "budget":budget,"seed":seed,"pred":out,"score":float(V(out)),
                      "tokens":int(tok),"expansions":int(exps),"wall_s":0.0})

            ctrl = LToTController(llm, LToTParams(), V, plateau_cfg=plateau_cfg, bars_cfg=bars_cfg,
                                  lambdas=lambdas, mainline_cfg=mainline_cfg, early_stop=True)
            out, tok, exps = ctrl.run(build_prompt(task,item), q, budget, task,
                                      scorer=score_expl, rng=random.Random(seed))
            aw.write({"kind":"run","task":task,"qid":item["qid"],"model":model_id,"method":"LToT",
                      "budget":budget,"seed":seed,"pred":out,"score":float(V(out)),
                      "tokens":int(tok),"expansions":int(exps),"wall_s":0.0})
    aw.close()
    print("[diag] wrote results/raw/diag_local.jsonl")

if __name__ == "__main__":
    main()
