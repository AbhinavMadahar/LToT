import argparse, os, json, random, glob, math, time
from pathlib import Path
import pandas as pd, yaml, numpy as np
from .inference.backends import make_llm
from .datasets import load_task
from .evaluators import exact_match, run_python_tests, eval_game24, subset_unit_tests
from .search.baselines import tot_baseline, mcts_pw_baseline
from .search.ltot_controller import LToTController
from .config import LToTParams
from .util.artifact_jsonl import ArtifactWriter
from .plotting.figures import main_equal_compute_figure
from .scorers.vlm import vlm_score
from .calibrate import calibrate as calibrate_equal
from statistics import median

def build_prompt(task, item):
    if task in ("gsm_plus","gsm_hard","math_500"):
        return lambda q: f"Think step by step and compute the final answer.\nQuestion: {q}\nAnswer:"
    elif task=="game24":
        return lambda q: f"Use (+,-,*,/) and parentheses to make 24 from digits {q}. Show steps, then final expression.\nAnswer:"
    elif task in ("humaneval","mbpp_lite"):
        return lambda q: f"{q}\n# Write the function implementation below.\n"
    else:
        return lambda q: f"{q}\nAnswer:"

def verifier_for(task, item):
    if task in ("gsm_plus","gsm_hard","math_500"):
        gold = item["answer"]
        return lambda text: 1.0 if exact_match(text.splitlines()[-1].strip(), gold) else 0.0
    elif task=="game24":
        def V(text):
            final = (text or "").strip().splitlines()[-1]
            return 1.0 if eval_game24(final) else 0.0
        return V
    elif task=="humaneval":
        return lambda text: 1.0 if run_python_tests(text, item["tests"]) else 0.0
    elif task=="mbpp_lite":
        tests_code = "".join((t + "\n") for t in (item.get("tests", [])))
        return lambda text: 1.0 if run_python_tests(text, tests_code) else 0.0
    return lambda text: 0.0

def scorer_for_exploration(task, noisy_cfg, item=None):
    if task == "humaneval":
        tests_src = item["tests"]
        def score(model, task, text):
            ok = run_python_tests(text, subset_unit_tests(tests_src, k=3))
            return (1.0 if ok else 0.0, 0)
        return score
    if task == "mbpp_lite":
        tests_src = "".join((t + "\n") for t in (item.get("tests", [])))
        def score(model, task, text):
            ok = run_python_tests(text, subset_unit_tests(tests_src, k=3))
            return (1.0 if ok else 0.0, 0)
        return score
    else:
        qtxt = (item.get("question") or item.get("prompt") or str(item.get("digits") or ""))
        if noisy_cfg.get("enabled", False):
            def score(model, task, text):
                return vlm_score(model, qtxt, text, (noisy_cfg.get("temp_low",0.0), noisy_cfg.get("temp_high",0.0)))
            return score
        else:
            # Zero-cost, deterministic proxy when noisy-v is disabled (keeps compute parity unchanged)
            def score(model, task, text):
                base = 0.5 + 0.5 * (len(str(text).strip()) > 0)
                return (max(0.0, min(1.0, base - 0.05)), 0)
            return score


def _load_cfg():
    with open("configs/experiments.yaml","r") as yf:
        return yaml.safe_load(yf)

def _mk_writer(path):
    return ArtifactWriter(path)

def _knobs(calib, task, method, default):
    return calib.get(task, {}).get(method, default)

def _main_grid(args, ycfg, llm, aw):
    plateau_cfg = ycfg.get("plateau", {})
    bars_cfg    = {**ycfg.get("bars", {}), "kappa":ycfg["controllers"]["ltot"]["kappa"], "delta":ycfg["controllers"]["ltot"]["delta"]}
    lambdas     = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
    mainline_cfg= {k: ycfg["controllers"]["ltot"][k] for k in ("mainline_topk","mainline_width","delta","initial_lateral_width","eta","b0","micro_probe","overflow_cap","confirmation_temp","micro_beam","beta_alpha","order_set") if k in ycfg["controllers"]["ltot"]}
    noisy_cfg   = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
    robust_cfg  = ycfg.get("robustness_study", {"enabled":False})
    parity      = ycfg.get("equal_compute", {})
    diag_cfg    = ycfg.get("diagnostics", {"enable":False})

    mainline_cfg["envelope_cfg"] = ycfg.get("envelope", None)
    mainline_cfg["dual_gate_cfg"] = ycfg.get("consistency",{}).get("qa_dual_gate", {"enabled": False})
    mainline_cfg["admission_gate_cfg"] = ycfg.get("consistency",{}).get("admission_gate", {"enabled": False})

    calib = {}
    if os.path.exists("results/calibration.json"):
        with open("results/calibration.json","r") as f:
            calib = json.load(f).get("knobs", {})

    rows = []

    for item in load_task(args.task, args.seed):
        q = item.get("question") or item.get("prompt") or str(item.get("digits"))
        V = verifier_for(args.task, item)
        for M in ["CoT","ToT","MCTS-PW","LToT"]:
            wall_start = time.monotonic()
            if M=="CoT":
                max_new = int(_knobs(calib,args.task,"CoT", args.budget))
                texts, toks = llm.generate([build_prompt(args.task,item)(q)], max_tokens=max(8, max_new))
                out = texts[0]; tokens = int(toks[0]); expansions = 1
            elif M=="ToT":
                beam = _knobs(calib,args.task,"ToT",5)
                score_expl = scorer_for_exploration(args.task, noisy_cfg if noisy_cfg.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
                out, tokens, expansions = tot_baseline(llm, args.task, q, budget_tokens=args.budget, beam=beam,
                                           exploration_scorer=score_expl, return_tokens=True,
                                           verifier=verifier_for(args.task, item), early_stop=False)
            elif M=="MCTS-PW":
                rolls = _knobs(calib,args.task,"MCTS-PW",64)
                score_expl = scorer_for_exploration(args.task, noisy_cfg if noisy_cfg.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
                out, tokens, expansions = mcts_pw_baseline(llm, args.task, q, budget_tokens=args.budget, rollouts=rolls,
                                               exploration_scorer=score_expl, return_tokens=True,
                                               verifier=verifier_for(args.task, item), early_stop=False)
            else:
                ml_cfg = dict(mainline_cfg)
                n0 = _knobs(calib,args.task,"LToT", ml_cfg.get("initial_lateral_width",128))
                ml_cfg["initial_lateral_width"] = int(n0)
                ml_cfg["envelope_cfg"] = ycfg.get("envelope", None)
                ml_cfg["noise_cfg"] = robust_cfg if robust_cfg.get("enabled", False) else {"enabled": False}
                ctrl = LToTController(llm, LToTParams(), V, plateau_cfg=ycfg.get("plateau",{}),
                                      bars_cfg=bars_cfg, lambdas=lambdas, mainline_cfg=ml_cfg,
                                      early_stop=False)
                score_expl = scorer_for_exploration(args.task, noisy_cfg if noisy_cfg.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
                def _log_cb(rec):
                    payload = {**rec}
                    payload.update({"task":args.task,"qid":item["qid"],"model":args.model,"method":"LToT",
                                    "budget":args.budget,"seed":args.seed})
                    aw.write(payload)
                text, tokens, rung_costs_all, _, expansions = ctrl.run(build_prompt(args.task,item), q, args.budget, args.task,
                                                                       scorer=score_expl, rng=random.Random(args.seed),
                                                                       return_logs=True, log_callback=_log_cb)
                out = text
                aw.write({"kind":"rung_costs","task":args.task,"qid":item["qid"],"model":args.model,
                          "method":"LToT","budget":args.budget,"seed":args.seed,"rung_costs": [int(x) for x in rung_costs_all]})
            v  = V(out)
            wall = time.monotonic() - wall_start
            rec = {"kind":"run","task":args.task,"qid":item["qid"],"model":args.model,
                   "method":M,"budget":args.budget,"seed":args.seed,"pred":out,"score":float(v),
                   "tokens": int(tokens), "expansions": int(expansions), "wall_s": float(wall)}
            if M=="LToT":
                rec["N0"] = int(n0)
            aw.write(rec); rows.append(rec)

            if M in ("ToT","MCTS-PW") and noisy_cfg.get("enabled", False) and (args.task in set(noisy_cfg.get("tasks", []))):
                from .scorers.vlm import vlm_score
                tau_v = float(ycfg.get("consistency",{}).get("qa_dual_gate",{}).get("tau_v", 0.85))
                s_vlm, s_cost = vlm_score(llm, "question", out, (noisy_cfg["temp_low"], noisy_cfg["temp_high"]))
                aw.write({"kind":"promotion_event","task":args.task,"qid":item["qid"],"model":args.model,"method":M,
                          "budget":args.budget,"seed":args.seed,"origin":"baseline_final",
                          "z": 0.0, "bar": tau_v, "proposed": bool(s_vlm >= tau_v), "accepted": bool(v >= 1.0), "rung": -1})

        if diag_cfg.get("enable", False):
            score_expl = scorer_for_exploration(args.task, noisy_cfg if noisy_cfg.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
            from .search.diagnostics import sh_only_lateralization, sh_on_mainlines
            out, tokens, expansions = sh_only_lateralization(llm, args.task, q, args.budget, score_expl,
                                                             initial_width=int(mainline_cfg.get("initial_lateral_width",128)),
                                                             eta=int(mainline_cfg.get("eta",4)), b0=int(mainline_cfg.get("b0",1)),
                                                             micro_beam=int(mainline_cfg.get("micro_beam",3)))
            v = V(out)
            r = {"kind":"run","task":args.task,"qid":item["qid"],"model":args.model,"method":"SH-LAT",
                 "budget":args.budget,"seed":args.seed,"pred":out,"score":float(v),
                 "tokens":int(tokens), "expansions":int(expansions), "wall_s":0.0}
            aw.write(r); rows.append(r)
            out, tokens, expansions = sh_on_mainlines(llm, args.task, q, args.budget, score_expl,
                                                      beam=int(mainline_cfg.get("mainline_width",2))*2,
                                                      eta=int(mainline_cfg.get("eta",4)), b0=int(mainline_cfg.get("b0",1)))
            v = V(out)
            r = {"kind":"run","task":args.task,"qid":item["qid"],"model":args.model,"method":"SH-MAIN",
                 "budget":args.budget,"seed":args.seed,"pred":out,"score":float(v),
                 "tokens":int(tokens), "expansions":int(expansions), "wall_s":0.0}
            aw.write(r); rows.append(r)
    return pd.DataFrame(rows)

def _width_scaling(args, ycfg, llm, aw):
    plate = ycfg.get("plateau", {})
    bars  = {**ycfg.get("bars", {}), "kappa":ycfg["controllers"]["ltot"]["kappa"], "delta":ycfg["controllers"]["ltot"]["delta"]}
    lamb  = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
    mlcfg = {k: ycfg["controllers"]["ltot"][k] for k in ("mainline_topk","mainline_width","delta","initial_lateral_width","eta","b0","micro_probe","overflow_cap","confirmation_temp","micro_beam","beta_alpha","order_set") if k in ycfg["controllers"]["ltot"]}
    noisy = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
    N0vals= ycfg.get("width_scaling",{}).get("N0_values",[32,64,128,256,512,1024])
    rows = []
    for item in load_task(args.task, args.seed):
        q = item.get("question") or item.get("prompt") or str(item.get("digits"))
        V = verifier_for(args.task, item)
        for N0 in N0vals:
            ml = dict(mlcfg); ml["initial_lateral_width"]=int(N0)
            ml["admission_gate_cfg"] = ycfg.get("consistency",{}).get("admission_gate", {"enabled": False})
            ctrl = LToTController(llm, LToTParams(), V, plateau_cfg=plate, bars_cfg=bars, lambdas=lamb, mainline_cfg=ml, early_stop=False)
            score_expl = scorer_for_exploration(args.task, noisy if noisy.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
            out, tokens, expansions = ctrl.run(build_prompt(args.task,item), q, args.budget, args.task, scorer=score_expl, rng=random.Random(args.seed))
            v = V(out)
            rec = {"kind":"width_scaling_run","task":args.task,"qid":item["qid"],"model":args.model,
                   "N0":int(N0),"budget":args.budget,"seed":args.seed,"score":float(v),"tokens":int(tokens),"expansions":int(expansions)}
            aw.write(rec); rows.append(rec)
    return pd.DataFrame(rows)

def _ablations(args, ycfg, llm, aw):
    plate = ycfg.get("plateau", {})
    bars  = {**ycfg.get("bars", {}), "kappa":ycfg["controllers"]["ltot"]["kappa"], "delta":ycfg["controllers"]["ltot"]["delta"]}
    lamb  = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
    mlcfg = {k: ycfg["controllers"]["ltot"][k] for k in ("mainline_topk","mainline_width","delta","initial_lateral_width","eta","b0","micro_probe","overflow_cap","confirmation_temp","micro_beam","beta_alpha","order_set") if k in ycfg["controllers"]["ltot"]}
    noisy = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
    abls  = ycfg.get("ablations", [])
    rows = []
    for item in load_task(args.task, args.seed):
        q = item.get("question") or item.get("prompt") or str(item.get("digits"))
        V = verifier_for(args.task, item)
        for ab in abls:
            mlx = dict(mlcfg); mlx["admission_gate_cfg"] = ycfg.get("consistency",{}).get("admission_gate", {"enabled": False})
            ctrl = LToTController(llm, LToTParams(), V, plateau_cfg=plate, bars_cfg=bars, lambdas=lamb, mainline_cfg=mlx, ablation=ab, early_stop=False)
            score_expl = scorer_for_exploration(args.task, noisy if noisy.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
            out, tokens, expansions = ctrl.run(build_prompt(args.task,item), q, args.budget, args.task, scorer=score_expl, rng=random.Random(args.seed))
            v = V(out)
            rec = {"kind":"ablation_run","task":args.task,"qid":item["qid"],"model":args.model,
                   "ablation":ab,"budget":args.budget,"seed":args.seed,"score":float(v),"tokens":int(tokens),"expansions":int(expansions)}
            aw.write(rec); rows.append(rec)
    return pd.DataFrame(rows)

def do_run(args):
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    ycfg = _load_cfg()
    plateau_cfg = ycfg.get("plateau", {})
    bars_cfg    = {**ycfg.get("bars", {}), "kappa":ycfg["controllers"]["ltot"]["kappa"], "delta":ycfg["controllers"]["ltot"]["delta"]}
    lambdas     = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
    mainline_cfg= {k: ycfg["controllers"]["ltot"][k] for k in ("mainline_topk","mainline_width","delta","initial_lateral_width","eta","b0","micro_probe","overflow_cap","confirmation_temp","micro_beam","beta_alpha","order_set") if k in ycfg["controllers"]["ltot"]}
    noisy_cfg   = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
    parity      = ycfg.get("equal_compute", {})

    aw = _mk_writer(args.out)
    rng = random.Random(args.seed)
    llm = make_llm(args.model)

    if parity.get("calibrate", True) and args.shard == 0:
        tasks_small = {}
        for t in ycfg["tasks"]:
            items = [it for i, it in zip(range(parity.get("sample_per_task",24)), load_task(t, args.seed))]
            tasks_small[t] = items
        state = calibrate_equal(
            llm, tasks_small, {"target": args.budget},
            methods=["CoT","ToT","MCTS-PW","LToT"],
            seed=args.seed,
            tol_pp=float(parity.get("tolerance_pp",2.0)),
            sample_per_task=int(parity.get("sample_per_task",24)),
            max_iters=int(parity.get("max_iters",4)),
        )
        aw.write({"kind":"calibration","state":state})

    df = _main_grid(args, ycfg, llm, aw)

    def mean_ci(series):
        x = np.asarray(series, float); n = len(x)
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1))/max(1, math.sqrt(n))
        return m, 1.96*se

    for (t, mth, mdl), g in df.groupby(["task","method","model"]):
        m, ci = mean_ci(g["score"])
        aw.write({"kind":"metric","task":t,"method":mth,"model":mdl,
                  "budget":args.budget,"seed":args.seed,
                  "metric":"Success@1" if t not in ("humaneval","mbpp_lite") else "Pass@1",
                  "score": m, "ci95": ci,
                  "median_tokens": float(np.median(g["tokens"]))})

    for (t, mdl), g in df.groupby(["task","model"]):
        target = float(args.budget)
        med_by_method = g.groupby("method")["tokens"].median().to_dict()
        for M, med in med_by_method.items():
            err = 100.0*(float(med)-target)/max(1.0,target)
            aw.write({"kind":"fairness","task":t,"model":mdl,"method":M,
                      "target_tokens":target,"median_tokens":float(med),
                      "error_pp": float(err)})

    svg = main_equal_compute_figure(df)
    aw.write({"kind":"figure_svg","name":"main_equal_compute","svg":svg})
    aw.close()

def do_widthscale(args):
    Path("results/raw_width").mkdir(parents=True, exist_ok=True)
    ycfg = _load_cfg()
    aw = _mk_writer(args.out)
    llm = make_llm(args.model)
    _width_scaling(args, ycfg, llm, aw)
    aw.close()

def do_ablate(args):
    Path("results/raw_ablate").mkdir(parents=True, exist_ok=True)
    ycfg = _load_cfg()
    aw = _mk_writer(args.out)
    llm = make_llm(args.model)
    _ablations(args, ycfg, llm, aw)
    aw.close()

def do_earlystop(args):
    Path("results/raw_latency").mkdir(parents=True, exist_ok=True)
    ycfg = _load_cfg()
    aw = _mk_writer(args.out)
    llm = make_llm(args.model)
    noisy = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
    for item in load_task(args.task, args.seed):
        q = item.get("question") or item.get("prompt") or str(item.get("digits"))
        V = verifier_for(args.task, item)
        plate = ycfg.get("plateau", {})
        bars  = {**ycfg.get("bars", {}), "kappa":ycfg["controllers"]["ltot"]["kappa"], "delta":ycfg["controllers"]["ltot"]["delta"]}
        lamb  = ycfg.get("consistency",{}).get("lambdas", {"logic":0.7,"syntax":0.2,"constraints":0.1})
        mlcfg = {k: ycfg["controllers"]["ltot"][k] for k in ("mainline_topk","mainline_width","delta","initial_lateral_width","eta","b0","micro_probe","overflow_cap","confirmation_temp","micro_beam","beta_alpha","order_set") if k in ycfg["controllers"]["ltot"]}
        mlcfg["admission_gate_cfg"] = ycfg.get("consistency",{}).get("admission_gate", {"enabled": False})
        score_expl = scorer_for_exploration(args.task, noisy if noisy.get("enabled",False) else {"temp_low":0.0,"temp_high":0.0}, item)
        w0 = time.monotonic(); texts, toks = llm.generate([build_prompt(args.task,item)(q)], max_tokens=max(8, args.budget//4)); w1 = time.monotonic()
        out = texts[0]; tok = int(toks[0]); hit = bool(V(out)>=1.0)
        aw.write({"kind":"earlystop_latency","task":args.task,"qid":item["qid"],"model":args.model,"method":"CoT",
                  "budget":args.budget,"seed":args.seed,"tokens":int(tok),"wall_s":float(w1-w0),"hit":hit,"expansions":1})
        w0 = time.monotonic(); out, tok, exps = tot_baseline(llm, args.task, q, args.budget, beam=5, exploration_scorer=score_expl, return_tokens=True, verifier=V, early_stop=True); w1=time.monotonic()
        aw.write({"kind":"earlystop_latency","task":args.task,"qid":item["qid"],"model":args.model,"method":"ToT",
                  "budget":args.budget,"seed":args.seed,"tokens":int(tok),"wall_s":float(w1-w0),"hit":bool(V(out)>=1.0),"expansions":int(exps)})
        w0 = time.monotonic(); out, tok, exps = mcts_pw_baseline(llm, args.task, q, args.budget, rollouts=64, exploration_scorer=score_expl, return_tokens=True, verifier=V, early_stop=True); w1=time.monotonic()
        aw.write({"kind":"earlystop_latency","task":args.task,"qid":item["qid"],"model":args.model,"method":"MCTS-PW",
                  "budget":args.budget,"seed":args.seed,"tokens":int(tok),"wall_s":float(w1-w0),"hit":bool(V(out)>=1.0),"expansions":int(exps)})
        ctrl = LToTController(llm, LToTParams(), V, plateau_cfg=plate, bars_cfg=bars, lambdas=lamb, mainline_cfg=mlcfg, early_stop=True)
        w0 = time.monotonic(); out, tok, exps = ctrl.run(build_prompt(args.task,item), q, args.budget, args.task, scorer=score_expl, rng=random.Random(args.seed)); w1=time.monotonic()
        aw.write({"kind":"earlystop_latency","task":args.task,"qid":item["qid"],"model":args.model,"method":"LToT",
                  "budget":args.budget,"seed":args.seed,"tokens":int(tok),"wall_s":float(w1-w0),"hit":bool(V(out)>=1.0),"expansions":int(exps)})
    aw.close()

def do_aggregate(args):
    writer = ArtifactWriter(args.artifact)
    run_rows = []; promo_rows = []; early_rows = []; ltot_runs = []
    rungcost_rows = []
    for path in sorted(glob.glob(os.path.join(args.inputs,"*.jsonl"))):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                writer.write(rec)
                if rec.get("kind") == "run":
                    run_rows.append(rec)
                    if rec.get("method")=="LToT" and "N0" in rec:
                        ltot_runs.append(rec)
                if rec.get("kind") == "promotion_event":
                    promo_rows.append(rec)
                if rec.get("kind") == "rung_costs":
                    rungcost_rows.append(rec)
    for path in sorted(glob.glob(os.path.join(args.inputs_width,"*.jsonl"))):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                writer.write(rec)
    for path in sorted(glob.glob(os.path.join(args.inputs_ablate,"*.jsonl"))):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                writer.write(rec)
    if getattr(args, "inputs_latency", None):
        for path in sorted(glob.glob(os.path.join(args.inputs_latency, "*.jsonl"))):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line); writer.write(rec)
                    if rec.get("kind")=="earlystop_latency": early_rows.append(rec)
    writer.close()


# --- Compute median expansions-to-first-verified (Option A metric) ---
import statistics as _stats
if early_rows:
    # keep only hits; for misses, we can treat as budget-capped (exclude from median as in paper)
    erdf = pd.DataFrame(early_rows)
    grp = erdf.groupby(["task","model","method","budget"], dropna=False)
    with open(args.artifact, "a", encoding="utf-8") as f:
        for keys, g in grp:
            # Filter to hits only; if none hit, skip to avoid degenerate medians
            gh = g[g["hit"]==True]
            if not gh.empty and "expansions" in gh.columns:
                med = float(_stats.median([int(x) for x in gh["expansions"].tolist()]))
                task, model, method, budget = keys
                f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                    "budget":budget,"metric":"median_expansions_to_first_verified",
                                    "score": med}) + "\n")

    if args.fig:
        if run_rows:
            df = pd.DataFrame(run_rows)
            svg = main_equal_compute_figure(df)
            os.makedirs(os.path.dirname(args.fig), exist_ok=True)
            with open(args.fig, "w", encoding="utf-8") as f:
                f.write(svg)

    if promo_rows:
        pdf = pd.DataFrame(promo_rows)
        grp = pdf.groupby(["task","model","method","budget","seed"], dropna=False)
        for keys, g in grp:
            proposed = int(g["proposed"].sum())
            accepted = int(g["accepted"].sum())
            fpr = float((proposed - accepted) / max(1, proposed))
            sel = float(accepted / max(1, proposed))
            task, model, method, budget, seed = keys
            with open(args.artifact, "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                    "budget":budget,"seed":seed,"metric":"false_promotion_rate",
                                    "score":fpr})+"\n")
                f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                    "budget":budget,"seed":seed,"metric":"promotion_selectivity",
                                    "score":sel})+"\n")

    if ltot_runs:
        ycfg = {}
        try:
            with open("configs/experiments.yaml","r") as yf: ycfg = yaml.safe_load(yf)
        except Exception:
            pass
        eta = float(ycfg.get("controllers",{}).get("ltot",{}).get("eta", 4))
        rdf = pd.DataFrame([r for r in run_rows if r.get("method")=="LToT" and "N0" in r and "expansions" in r])
        if not rdf.empty:
            X = np.asarray([r["N0"] * (math.log(max(2, r["N0"])) / math.log(eta)) for _, r in rdf.iterrows()], float)
            Y = np.asarray([r["expansions"] for _, r in rdf.iterrows()], float)
            if len(X) >= 2 and np.var(X) > 0:
                A = np.vstack([X, np.ones_like(X)]).T
                a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
                Yhat = a*X + b
                ss_res = float(np.sum((Y - Yhat)**2))
                ss_tot = float(np.sum((Y - np.mean(Y))**2))
                R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                with open(args.artifact, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"kind":"metric","task":"_pooled_","model":"_all_","method":"LToT",
                                         "budget":"_pooled_","metric":"cost_fit_R2_expansions",
                                         "score":float(R2)})+"\n")
        if rungcost_rows:
            import statistics as _stats
            groups = {}
            for rc in rungcost_rows:
                key = (rc["task"], rc["model"], rc.get("method","LToT"), rc["budget"], rc["seed"])
                groups.setdefault(key, []).append(rc.get("rung_costs", []))
            with open(args.artifact, "a", encoding="utf-8") as f:
                for (task, model, method, budget, seed), arrays in groups.items():
                    cvs, nrungs = [], []
                    for arr in arrays:
                        arr = [float(x) for x in (arr or [])]
                        if len(arr) >= 2 and (sum(arr) > 0):
                            m = _stats.mean(arr)
                            s = _stats.pstdev(arr)
                            cvs.append( (s / m) if m > 0 else 0.0 )
                            nrungs.append(len(arr))
                    if cvs:
                        cv_mean = float(_stats.mean(cvs))
                        nr_mean = float(_stats.mean(nrungs)) if nrungs else 0.0
                        nr_sd   = float(_stats.pstdev(nrungs)) if len(nrungs) >= 2 else 0.0
                        f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                             "budget":budget,"metric":"rung_cost_cv_mean","score":cv_mean})+"\n")
                        f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                             "budget":budget,"metric":"num_rungs_mean","score":nr_mean})+"\n")
                        f.write(json.dumps({"kind":"metric","task":task,"model":model,"method":method,
                                             "budget":budget,"metric":"num_rungs_sd","score":nr_sd})+"\n")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--budget", type=int, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shard", type=int, default=0)

    w = sub.add_parser("widthscale")
    w.add_argument("--model", required=True)
    w.add_argument("--task", required=True)
    w.add_argument("--budget", type=int, required=True)
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--shard", type=int, default=0)

    a = sub.add_parser("ablate")
    a.add_argument("--model", required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--budget", type=int, required=True)
    a.add_argument("--seed", type=int, required=True)   # <-- fixed (type=int)
    a.add_argument("--out", required=True)
    a.add_argument("--shard", type=int, default=0)      # <-- fixed (type=int)

    g = sub.add_parser("aggregate")
    g.add_argument("--inputs", required=True)
    g.add_argument("--inputs_width", required=True)
    g.add_argument("--inputs_ablate", required=True)
    g.add_argument("--inputs_latency", required=False, default=None)
    g.add_argument("--artifact", required=True)
    g.add_argument("--fig", required=True)

    es = sub.add_parser("earlystop")
    es.add_argument("--model", required=True)
    es.add_argument("--task", required=True)
    es.add_argument("--budget", type=int, required=True)
    es.add_argument("--seed", type=int, required=True)
    es.add_argument("--out", required=True)
    es.add_argument("--shard", type=int, default=0)

    args = ap.parse_args()
    if args.cmd=="run": do_run(args)
    elif args.cmd=="widthscale": do_widthscale(args)
    elif args.cmd=="ablate": do_ablate(args)
    elif args.cmd=="earlystop": do_earlystop(args)
    else: do_aggregate(args)

if __name__=="__main__":
    main()
