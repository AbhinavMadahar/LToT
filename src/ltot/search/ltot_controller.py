from typing import Dict, Any, List, Tuple, Callable
import random, math, numpy as np
from .lrsc import LRSC, BranchState
from ..scorers.consistency import c_local_score

class Branch:
    def __init__(self, text:str, v_hist:List[float], h_hist:List[int], origin:str):
        self.text=text; self.v_hist=v_hist; self.h_hist=h_hist; self.origin=origin

class LToTController:
    def __init__(self, model, params, verifier, plateau_cfg, bars_cfg, lambdas, mainline_cfg,
                 ablation: str = None, early_stop: bool = False):
        self.model = model
        self.verifier = verifier
        self.plateau_cfg = plateau_cfg
        self.bar_cfg = bars_cfg
        self.lambdas = lambdas
        self.mainline_cfg = mainline_cfg
        self.admission_cfg = dict(mainline_cfg.get("admission_gate_cfg", {"enabled": False}))
        self.ablation = ablation
        self.early_stop = early_stop
        self.progress_ewma = 0.0
        self.progress_lastC = 0
        self.progress_pat = 0
        self.frozen_survivors: List[BranchState] = []

    def run(self, prompt_builder, q, budget_tokens: int, task:str, scorer, rng,
            return_logs: bool=False, log_callback=None):
        tokens_spent = 0
        expansions_total = 0
        mainlines = []
        txts, toks = self.model.generate([prompt_builder(q)], max_tokens=max(32, budget_tokens//8))
        tokens_spent += int(toks[0]); expansions_total += 1
        mainlines.append(Branch(text=txts[0], v_hist=[self.verifier(txts[0])], h_hist=[0], origin="MAIN"))
        Bt = max(b.v_hist[-1] for b in mainlines)

        proposals = []
        rung_costs_all = []

        def plateau_ok(deltaC):
            if self.ablation == "no_plateau":
                return False
            beta = self.plateau_cfg.get("ewma_beta", 0.3)
            tau = self.plateau_cfg.get("tau", 1e-4)
            hyster = self.plateau_cfg.get("hysteresis", 5e-5)
            minC = self.plateau_cfg.get("min_compute_delta", 8)
            if deltaC < minC: return True
            inc = (max(b.v_hist[-1] for b in mainlines) - Bt) / max(1, deltaC)
            self.progress_ewma = (beta)*inc + (1-beta)*self.progress_ewma
            return self.progress_ewma >= (tau - hyster)

        while tokens_spent < budget_tokens:
            started_C = tokens_spent
            steps = 0
            while tokens_spent < budget_tokens:
                outs, toks = self.model.generate([mainlines[0].text + "\nContinue:"], max_tokens=max(24, budget_tokens//64))
                tokens_spent += int(toks[0]); expansions_total += 1
                mainlines[0].text += "\n" + outs[0]
                vnow = self.verifier(mainlines[0].text)
                mainlines[0].v_hist.append(vnow)
                mainlines[0].h_hist.append(mainlines[0].h_hist[-1]+1)
                newBt = max(b.v_hist[-1] for b in mainlines)
                dBt = newBt - Bt
                Bt = newBt
                steps += 1
                deltaC = tokens_spent - started_C
                inc = dBt / max(1, deltaC)
                beta = self.plateau_cfg.get("ewma_beta",0.3)
                self.progress_ewma = beta*inc + (1-beta)*self.progress_ewma
                if self.early_stop and vnow >= 1.0:
                    if return_logs:
                        return (mainlines[0].text, tokens_spent, rung_costs_all, proposals, expansions_total)
                    return (mainlines[0].text, tokens_spent, expansions_total)
                if not plateau_ok(deltaC):
                    break

            initial_width = int(self.mainline_cfg.get("initial_lateral_width", 128))
            laterals = self.frozen_survivors[:]; self.frozen_survivors = []
            for _ in range(max(0, initial_width - len(laterals))):
                outs, toks = self.model.generate(
                    [mainlines[0].text + "\nConsider a logically different path:"],
                    max_tokens=max(32, budget_tokens//96), temperature=0.8, top_p=0.95
                )
                cand = outs[0].strip()
                tokens_spent += int(toks[0]); expansions_total += 1
                admit = True; cval = None; ccost = 0
                if self.admission_cfg.get("enabled", False):
                    # Admission gate: c_local >= tau_c (+ tightening if LM-only)
                    t_c = float(self.admission_cfg.get("temperature", 0.0))
                    cval, ccost = c_local_score(self.model, mainlines[0].text, cand, temperature=t_c)
                    tokens_spent += int(ccost)
                    tau_c = float(self.admission_cfg.get("tau_c", 0.75))
                    # Tighten if syntax/constraints are effectively absent and logic carries weight
                    if (self.lambdas.get("syntax", 0.0) + self.lambdas.get("constraints", 0.0)) <= 1e-6 \
                       and self.lambdas.get("logic", 0.0) >= 0.7:
                        tau_c += float(self.admission_cfg.get("tighten_if_lmonly", 0.10))
                    admit = (float(cval) >= tau_c)
                    if log_callback:
                        log_callback({"kind":"admission_event","c": float(cval), "tau_c": float(tau_c), "accepted": bool(admit)})
                if admit:
                    laterals.append(BranchState(root_text=cand, parent_text=mainlines[0].text, origin="LAT"))


            ab = self.ablation or ""
            lrsc = LRSC(
                self.model, scorer, self.bar_cfg,
                eta=self.mainline_cfg.get("eta",4),
                b0=self.mainline_cfg.get("b0",1),
                micro_probe=self.mainline_cfg.get("micro_probe",1),
                overflow_cap=(0.0 if ab=="overflow_off" else self.mainline_cfg.get("overflow_cap",0.15)),
                order_set=((1,) if ab=="no_curvature" else tuple(self.mainline_cfg.get("order_set",[1,2]))),
                use_width_bar=(ab!="no_width_bar"),
                allow_short_circuit=(ab!="no_short_circuit"),
                require_confirm=(ab!="no_confirm"),
                confirm_temps=tuple(self.mainline_cfg.get("confirmation_temp",[0.7,0.95])),
                micro_beam=int(self.mainline_cfg.get("micro_beam",3)),
                beta_alpha=float(self.mainline_cfg.get("beta_alpha",0.5)),
                envelope_cfg=self.mainline_cfg.get("envelope_cfg", None),
                noise_cfg=self.mainline_cfg.get("noise_cfg", None),
                dual_gate_cfg=self.mainline_cfg.get("dual_gate_cfg", None)
            )
            promoted, survivors, spend, rung_costs, rung_expansions, exp_add = lrsc.run(
                laterals, Bt, task, budget_tokens - tokens_spent, self.lambdas, rng, verifier=self.verifier,
                log_callback=(lambda rec: log_callback(rec) if log_callback else None)
            )
            tokens_spent += spend; expansions_total += int(exp_add)
            rung_costs_all.extend(rung_costs)
            if return_logs and log_callback:
                log_callback({"kind":"rung_expansions","rung_expansions":[int(x) for x in rung_expansions]})
            self.frozen_survivors = survivors
            if promoted:
                mainlines = [Branch(text=promoted, v_hist=[self.verifier(promoted)], h_hist=[0], origin="MAIN")]
                Bt = max(b.v_hist[-1] for b in mainlines)
            if tokens_spent >= budget_tokens:
                break

        best = max(mainlines, key=lambda b: self.verifier(b.text)) if mainlines else None
        if return_logs:
            return (best.text if best else "", tokens_spent, rung_costs_all, proposals, expansions_total)
        return (best.text if best else "", tokens_spent, expansions_total)
