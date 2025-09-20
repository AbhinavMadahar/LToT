def dry_run_tokens(llm, method, task, item, knob_val, target_budget):
    from .search.baselines import tot_baseline, mcts_pw_baseline
    from .search.ltot_controller import LToTController
    from .config import LToTParams
    from .evaluators import exact_match, run_python_tests
    from .run import build_prompt, verifier_for, scorer_for_exploration
    import yaml, random

    q = item.get("question") or item.get("prompt") or str(item.get("digits"))
    V = verifier_for(task, item)

    if method=="ToT":
        _, tokens, _ = tot_baseline(llm, task, q, budget_tokens=target_budget, beam=knob_val, return_tokens=True)
        return tokens
    if method=="MCTS-PW":
        _, tokens, _ = mcts_pw_baseline(llm, task, q, budget_tokens=target_budget, rollouts=knob_val, return_tokens=True)
        return tokens
    if method=="LToT":
        params = LToTParams()
        import yaml
        with open("configs/experiments.yaml","r") as yf:
            ycfg = yaml.safe_load(yf)
        bars_cfg = {**ycfg.get("bars", {}), "kappa": ycfg["controllers"]["ltot"]["kappa"], "delta": ycfg["controllers"]["ltot"]["delta"]}
        plateau_cfg = ycfg.get("plateau", {})
        lambdas = ycfg.get("consistency",{}).get("lambdas",{"logic":0.7,"syntax":0.2,"constraints":0.1})
        mainline_cfg = dict(ycfg["controllers"]["ltot"]); mainline_cfg["initial_lateral_width"]=int(knob_val)
        ctrl = LToTController(llm, params, V, plateau_cfg=plateau_cfg, bars_cfg=bars_cfg, lambdas=lambdas, mainline_cfg=mainline_cfg, early_stop=False)
        noisy = ycfg.get("noisy_v_study", {"enabled":False,"temp_low":0.0,"temp_high":0.0})
        score_expl = scorer_for_exploration(task, noisy, item)
        _, tokens, _ = ctrl.run(build_prompt(task,item), q, target_budget, task, scorer=score_expl, rng=random.Random(0))
        return tokens
    if method=="CoT":
        texts, toks = llm.generate([build_prompt(task,item)(q)], max_tokens=int(knob_val))
        return int(toks[0])
    return target_budget
