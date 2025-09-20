from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import math, random
from ..scorers.consistency import c_local_score

def robust_z(vals, winsor=None):
    x = np.asarray(vals, float)
    if winsor:
        lo, hi = np.percentile(x, [100*winsor/1000.0, 100-100*winsor/1000.0])
        x = np.clip(x, lo, hi)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad), med, 1.4826*mad

def effective_width_from_corr(z_pred):
    x = np.asarray(z_pred, float)
    if len(x) < 2: return max(1, len(x))
    diffs = np.diff(np.sort(x))
    v = np.var(diffs) + 1e-8
    rho = 1.0/(1.0+10.0*v)
    return int(max(1, round((1-rho) * len(x))))

def width_aware_bar(n_eff: int, m_orders: int, bar_cfg: Dict[str,Any]) -> float:
    tail   = bar_cfg.get("tail_model", "subgaussian")
    kappa  = float(bar_cfg.get("kappa", 1.0))
    delta  = float(bar_cfg.get("delta", 0.0))
    neffM  = max(2, int(max(1, n_eff)) * max(1, int(m_orders)))
    if tail == "subgaussian":
        return kappa * math.sqrt(2.0 * math.log(neffM)) + delta
    if tail == "subgamma":
        nu = float(bar_cfg.get("subgamma",{}).get("nu",1.0))
        c  = float(bar_cfg.get("subgamma",{}).get("c",1.0))
        return kappa * (math.sqrt(2.0*nu*math.log(neffM)) + c*math.log(neffM)) + delta
    if tail == "subweibull":
        K = float(bar_cfg.get("subweibull",{}).get("K",1.0))
        alpha = float(bar_cfg.get("subweibull",{}).get("alpha",1.5))
        return K * (math.log(neffM) ** (1.0/max(alpha,1e-6))) + delta
    return kappa * math.sqrt(2.0 * math.log(neffM)) + delta

def _poly_forecast(C_hist, Vtil, orders=(1,2)):
    gains = {}
    x = np.asarray(C_hist[-4:], float)
    y = np.asarray(Vtil[-4:], float)
    if len(x) < 2: return gains
    x = x - x.min()
    for m in orders:
        deg = min(m, len(x)-1)
        if deg <= 0: continue
        coef = np.polyfit(x, y, deg=deg)
        p = np.poly1d(coef); dp = np.polyder(p, 1)
        gains[m] = float(dp(x[-1]))
    return gains

class BranchState:
    def __init__(self, root_text:str, parent_text:str, origin:str):
        self.root_text = root_text
        self.parent_text = parent_text
        self.origin = origin
        self.tokens_spent = 0
        self.envelope_hist: List[Tuple[int, float, float, float]] = []
        self.accum_text = (parent_text + "\n" + root_text).strip()

class LRSC:
    def __init__(self, model, scorer, bar_cfg, eta:int, b0:int, micro_probe:int, overflow_cap:float,
                 order_set:Tuple[int,...]=(1,2), use_width_bar:bool=True, allow_short_circuit:bool=True,
                 require_confirm:bool=True, confirm_temps:Tuple[float,float]=(0.7,0.95),
                 micro_beam:int=3, beta_alpha:float=0.5, envelope_cfg=None, noise_cfg=None,
                 dual_gate_cfg: Dict[str,Any] = None):
        self.model = model
        self.scorer = scorer
        self.bar_cfg = bar_cfg
        self.eta = eta
        self.b0 = b0
        self.micro_probe = micro_probe
        self.overflow_cap = overflow_cap
        self.order_set = tuple(int(m) for m in (order_set or (1,)))
        self.use_width_bar = use_width_bar
        self.allow_short_circuit = allow_short_circuit
        self.require_confirm = bool(require_confirm)
        self.confirm_temps = confirm_temps
        self.micro_beam = max(1, int(micro_beam))
        self.beta_alpha = float(beta_alpha)
        self.envelope_cfg = envelope_cfg or {"agg":"topk"}
        self.noise_cfg = noise_cfg or {"enabled": False}
        self.dual = (dual_gate_cfg or {"enabled": False, "tau_c": 0.75, "q": 0.25})

    def run(self, laterals: List[BranchState], Bt: float, task: str, budget_tokens: int,
            lambdas: Dict[str,float], rng, verifier: Callable[[str], float], log_callback=None):
        def log(rec):
            if log_callback: log_callback(rec)

        def _aggregate(scores):
            if not scores: return 0.0, float(self.micro_beam)
            agg = (self.envelope_cfg.get("agg") or "topk").lower()
            xs = np.asarray(scores, float)
            if agg == "trimmed":
                f = float(self.envelope_cfg.get("trim_frac", 0.15))
                n = len(xs); k = int(max(0, min(n//2, round(f*n))))
                if 2*k >= n: return float(xs.mean()), float(min(n, self.micro_beam))
                x = np.sort(xs)[k: n-k]; Keff = float(len(x))
                return float(x.mean()), Keff
            if agg == "power":
                p = float(self.envelope_cfg.get("power_p", 1.5))
                x = np.maximum(xs, 1e-8)**p; Keff = float(len(xs))
                return float((x.mean())**(1.0/p)), Keff
            if agg == "weighted":
                beta = 5.0
                w = np.exp(beta*(xs - xs.max()))
                w = w / max(1e-8, w.sum())
                wmax = float(self.envelope_cfg.get("omega_max", 0.7))
                w = np.minimum(w, wmax); w = w / max(1e-8, w.sum())
                Keff = 1.0/float(np.sum(w*w) + 1e-8)
                return float(np.sum(w*xs)), float(Keff)
            k = max(1, min(self.micro_beam, len(xs)))
            x = np.mean(np.sort(xs)[-k:])
            return float(x), float(k)

        def _beta_smooth(v, k_eff):
            a = self.beta_alpha
            return float((k_eff * v + a) / (k_eff + 2*a))

        def _ht_noise(scale: float, tail: str) -> float:
            t = (tail or "laplace").lower()
            if t == "studentt":
                return float(np.random.standard_t(df=2)) * scale
            return float(np.random.laplace(0.0, scale))

        def _probe_once(br: BranchState, prompt_suffix:str, max_tokens:int, temperature:float=0.8, top_p:float=0.95):
            prompts = [br.accum_text + "\n" + prompt_suffix for _ in range(self.micro_beam)]
            outs, toks = self.model.generate(prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            tcost = int(sum(toks)); scores=[]
            for o in outs:
                s, vcost = self.scorer(self.model, task, o); tcost += int(vcost)
                if self.noise_cfg.get("enabled", False):
                    tail  = str(self.noise_cfg.get("tail", "laplace"))
                    scale = float(self.noise_cfg.get("scale", 0.15))
                    rho   = float(self.noise_cfg.get("corr_rho", 0.20))
                    xi    = _ht_noise(scale, tail)
                    s = s + rho * rung_shared + math.sqrt(max(0.0, 1.0 - rho*rho)) * xi
                    s = max(0.0, min(1.0, float(s)))
                scores.append(float(s))
            return outs, scores, tcost, len(outs)

        def _update_envelope(br: BranchState, scores: List[float], tcost: int):
            V, Keff = _aggregate(scores)
            Vt = _beta_smooth(V, float(Keff))
            h  = (br.envelope_hist[-1][0] + 1) if br.envelope_hist else 1
            C  = (br.envelope_hist[-1][1] if br.envelope_hist else 0.0) + float(tcost)
            br.envelope_hist.append((h, C, V, Vt))
            return V, Vt

        def _with_ctx(br: BranchState, s: str) -> str:
            return (br.accum_text + "\n" + s).strip()

        def _any_verified(br: BranchState, seq):
            for s in seq:
                if verifier(_with_ctx(br, s)) >= 1.0:
                    return True
            return False

        def _first_verified_text(br: BranchState, seq, fallback: str) -> str:
            for s in seq:
                if verifier(_with_ctx(br, s)) >= 1.0:
                    return _with_ctx(br, s)
            return _with_ctx(br, fallback)

        def _passes_consistency_gate(parent_text: str, steps: List[str]) -> Tuple[bool, float, int]:
            if not self.dual.get("enabled", False) or not steps:
                return True, 1.0, 0
            q = float(self.dual.get("q", 0.25))
            scores, cost = [], 0
            for s in steps:
                c, ccost = c_local_score(self.model, parent_text, s, temperature=0.0)
                scores.append(float(c)); cost += int(ccost)
            if not scores:
                return True, 1.0, cost
            scores = np.asarray(scores, float)
            cq = float(np.quantile(scores, q))
            return (cq >= float(self.dual.get("tau_c", 0.75))), cq, cost

        tokens_spent = 0
        rung_costs = []
        Sr = list(laterals)
        r = 0
        promoted_text = None
        survivors: List[BranchState] = []
        initial_n = len(Sr) if Sr else 1
        total_expansions = 0
        rung_expansions = []
        rung_shared = 0.0

        while Sr and tokens_spent < budget_tokens:
            before = tokens_spent
            expansions_this_rung = 0
            if self.noise_cfg.get("enabled", False):
                rung_shared = _ht_noise(float(self.noise_cfg.get("scale", 0.15)), str(self.noise_cfg.get("tail", "laplace")))

            denom = max(8, initial_n * (self.eta ** max(0, r)) * 4)
            unit_tokens  = max(16, int((budget_tokens - tokens_spent) / denom))
            full_tokens  = max(24, int(self.b0 * (self.eta ** r) * unit_tokens))
            micro_tokens = max(16, int(self.micro_probe * unit_tokens))

            for br in Sr:
                if not br.envelope_hist:
                    outs, scores, tcost, nexp = _probe_once(br, "Continue one step:", max_tokens=micro_tokens)
                    tokens_spent += tcost; br.tokens_spent += tcost
                    expansions_this_rung += int(nexp); total_expansions += int(nexp)
                    _update_envelope(br, scores, tcost)

            z_pred = []
            C_hist, Vt_hist = {}, {}
            for br in Sr:
                C_hist[br] = [C for (h,C,V,Vt) in br.envelope_hist][-4:] if br.envelope_hist else []
                Vt_hist[br]= [Vt for (h,C,V,Vt) in br.envelope_hist][-4:] if br.envelope_hist else []

            for br in Sr:
                orders = tuple(self.order_set) if self.order_set else (1,)
                gains = _poly_forecast(C_hist.get(br,[]), Vt_hist.get(br,[]), orders=orders)
                z_pred.append(max(gains.values()) if gains else -1e9)

            z, mu, s = robust_z(z_pred, winsor=self.bar_cfg.get("winsorize_z", None))
            n_eff = effective_width_from_corr(z_pred) if self.use_width_bar else len(Sr)
            m_orders = len(self.order_set) if self.order_set else 1
            bar = width_aware_bar(n_eff, m_orders, self.bar_cfg)

            order = np.argsort(-z)
            Qr = max(1, len(Sr)//max(2, self.eta))
            keep_idx = order[:Qr]
            overflow_budget = int(self.overflow_cap * len(Sr))
            overflow_idx = [i for i in order[Qr:Qr+overflow_budget] if (z[i] >= bar)]

            next_S = []
            for idx in keep_idx:
                br = Sr[idx]
                outs, scores, tcost, nexp = _probe_once(br, "Continue one step:", max_tokens=full_tokens)
                tokens_spent += tcost; br.tokens_spent += tcost
                expansions_this_rung += int(nexp); total_expansions += int(nexp)
                V, Vt = _update_envelope(br, scores, tcost)
                if self.allow_short_circuit and (V >= (Bt + float(self.bar_cfg.get("delta",0.0)))):
                    proposed_ok = _any_verified(br, outs)
                    accepted = False
                    outs2 = []
                    if proposed_ok:
                        if self.require_confirm:
                            t_lo, t_hi = self.confirm_temps
                            ctemp = random.uniform(t_lo, t_hi)
                            outs2, _, tcost2, nexp2 = _probe_once(
                                br, "Re-derive the final step succinctly:", max_tokens=micro_tokens,
                                temperature=ctemp, top_p=0.9
                            )
                            tokens_spent += tcost2; br.tokens_spent += tcost2
                            expansions_this_rung += int(nexp2); total_expansions += int(nexp2)
                            accepted = _any_verified(br, outs2)
                        else:
                            accepted = True
                    if accepted:
                        steps_for_gate = (outs2 if (outs2 and self.require_confirm) else outs)
                        ok_c, cval, ccost = _passes_consistency_gate(br.accum_text, steps_for_gate)
                        tokens_spent += ccost; br.tokens_spent += ccost
                        accepted = accepted and ok_c
                    log({"kind":"promotion_event","origin":"keeper","z":float(z[idx]),"bar":float(bar),
                         "proposed": bool(proposed_ok), "accepted": bool(accepted), "rung": int(r)})
                    if accepted:
                        promoted_text = _first_verified_text(br, outs2 if outs2 else outs, fallback=(outs2[0] if outs2 else outs[0]))
                        break
                next_S.append(br)
            if promoted_text:
                rung_costs.append(tokens_spent - before)
                rung_expansions.append(int(expansions_this_rung))
                break

            for idx in overflow_idx:
                br = Sr[idx]
                outs, scores, tcost, nexp = _probe_once(
                    br, "Re-evaluate via an alternative step order:",
                    max_tokens=micro_tokens, temperature=0.9, top_p=0.9
                )
                tokens_spent += tcost; br.tokens_spent += tcost
                expansions_this_rung += int(nexp); total_expansions += int(nexp)
                V, Vt = _update_envelope(br, scores, tcost)
                outs_c, scores_c, tcost_c, nexp_c = [], [], 0, 0
                Vc = Vt
                if self.require_confirm:
                    t_lo, t_hi = self.confirm_temps
                    outs_c, scores_c, tcost_c, nexp_c = _probe_once(
                        br, "Independent confirmation of the previous step:",
                        max_tokens=micro_tokens, temperature=random.uniform(t_lo,t_hi), top_p=0.9
                    )
                    tokens_spent += tcost_c; br.tokens_spent += tcost_c
                    expansions_this_rung += int(nexp_c); total_expansions += int(nexp_c)
                    Vc, _ = _update_envelope(br, scores_c, tcost_c)

                orders = tuple(self.order_set) if self.order_set else (1,)
                C_hist_conf = [C for (h,C,_,_) in br.envelope_hist][-4:]
                Vt_hist_conf= [Vt for (h,_,_,Vt) in br.envelope_hist][-4:]
                gains_conf   = _poly_forecast(C_hist_conf, Vt_hist_conf, orders=orders)
                z_conf = (max(gains_conf.values()) - mu)/(s + 1e-8) if gains_conf else -1e9

                promoted_here = False
                if self.allow_short_circuit and (Vc >= (Bt + float(self.bar_cfg.get("delta",0.0)))):
                    if self.require_confirm:
                        proposed_ok = _any_verified(br, outs_c)
                        accepted = proposed_ok
                        used_seq = outs_c
                    else:
                        proposed_ok = _any_verified(br, outs)
                        accepted = proposed_ok
                        used_seq = outs
                    if accepted:
                        ok_c, cval, ccost = _passes_consistency_gate(br.accum_text, used_seq)
                        tokens_spent += ccost; br.tokens_spent += ccost
                        accepted = accepted and ok_c
                    log({"kind":"promotion_event","origin":"overflow","z":float(z[idx]),"bar":float(bar),
                         "proposed": bool(proposed_ok), "accepted": bool(accepted), "rung": int(r)})
                    if accepted:
                        promoted_text = _first_verified_text(br, used_seq, fallback=(used_seq[0] if used_seq else (outs[0] if outs else br.accum_text)))
                        promoted_here = True
                if promoted_here:
                    next_S.append(br); 

                keep_overflow = (z_conf >= bar)
                log({"kind":"overflow_survivor","rung":int(r),"z_conf":float(z_conf),"bar":float(bar),
                     "kept": bool(keep_overflow)})
                if keep_overflow:
                    next_S.append(br)

            rung_costs.append(tokens_spent - before)
            rung_expansions.append(int(expansions_this_rung))
            log({"kind":"lrsc_rung","rung":int(r),"n_before":int(len(Sr)),"n_survive":int(len(next_S)),
                 "bar":float(bar),"tokens_delta":int(rung_costs[-1])})
            log({"kind":"rung_expansions","rung_expansions":[int(expansions_this_rung)]})
            Sr = next_S
            r += 1

        if promoted_text:
            return promoted_text, Sr, tokens_spent, rung_costs, rung_expansions, total_expansions
        return None, Sr, tokens_spent, rung_costs, rung_expansions, total_expansions
