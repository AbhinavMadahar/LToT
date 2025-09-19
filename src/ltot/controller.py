
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math, time, statistics as stats
import numpy as np

@dataclass
class BranchEnvelope:
    topk: int = 3
    alpha: float = 0.5
    history: List[Tuple[int, float]] = field(default_factory=list)

    def update(self, compute: int, leaf_scores: List[float]) -> float:
        if not leaf_scores:
            v = 0.0
        else:
            arr = sorted(leaf_scores, reverse=True)[: self.topk]
            v = float(sum(arr)) / len(arr)
        Kstar = min(self.topk, len(leaf_scores)) if leaf_scores else 0
        v_s = (Kstar * v + self.alpha) / (Kstar + 2 * self.alpha) if Kstar > 0 else 0.5
        self.history.append((compute, v_s))
        return v_s

    def improvement(self, order: int = 1) -> float:
        if len(self.history) < 2:
            return 0.0
        pts = self.history[-3:]
        xs = np.array([c for c, _ in pts], dtype=float)
        ys = np.array([v for _, v in pts], dtype=float)
        if np.all(xs == xs[0]):
            return 0.0
        deg = min(order, len(pts) - 1)
        coeffs = np.polyfit(xs, ys, deg=deg)
        d = np.polyder(coeffs, m=order)
        return float(np.polyval(d, xs[-1]))

@dataclass
class PlateauDetector:
    alpha: float = 0.3
    slope_threshold: float = 1e-4
    min_window: int = 200
    ewma: Optional[float] = None
    last_compute: int = 0

    def update(self, compute: int, delta_bar: float):
        if compute - self.last_compute < self.min_window:
            return False
        self.last_compute = compute
        self.ewma = (self.alpha * delta_bar + (1 - self.alpha) * (self.ewma or delta_bar))
        return (self.ewma or 0.0) < self.slope_threshold

@dataclass
class LateralRacing:
    eta: int = 4
    base_budget: int = 1
    micro_probe: int = 1
    overflow_cap: float = 0.15
    bar_kappa: float = 1.0
    bar_delta: float = 0.05
    derivatives: Tuple[int, ...] = (1, 2)

    def width_aware_bar(self, width: int) -> float:
        return self.bar_kappa * math.sqrt(2.0 * max(1, math.log(max(2, width)))) + self.bar_delta

    def standardized_gain(self, zs: List[float]) -> float:
        if not zs:
            return -1e9
        m = stats.median(zs)
        mad = stats.median([abs(z - m) for z in zs]) or 1e-6
        return (zs[-1] - m) / (1.4826 * mad)

@dataclass
class LToTController:
    envelope_topk: int = 3
    horizons: List[int] = field(default_factory=lambda: [1,2,4,8,16,32,64])
    lrs: LateralRacing = field(default_factory=LateralRacing)
    plateau: PlateauDetector = field(default_factory=PlateauDetector)

    mainline_bar: float = 0.0
    compute_spent: int = 0

    def run_lateral_phase(self, lateral_branches: List[Dict[str, Any]]):
        survivors = lateral_branches[:]
        promoted = None

        horizon_idx = 0
        while survivors and horizon_idx < len(self.horizons):
            width = len(survivors)
            bar = self.lrs.width_aware_bar(width)
            for b in survivors:
                env: BranchEnvelope = b.setdefault("_env", BranchEnvelope(self.envelope_topk))
                leaves = b.get("leaves_cb", lambda h: [0.0])(self.horizons[horizon_idx])
                vtilde = env.update(self.compute_spent, leaves)
                zs = []
                for d in self.lrs.derivatives:
                    zs.append(env.improvement(order=d))
                z = self.lrs.standardized_gain(zs)
                b["_z"] = z
                b["_v"] = vtilde

            keep = max(1, len(survivors) // self.lrs.eta)
            survivors.sort(key=lambda x: x["_z"], reverse=True)
            keep_set = survivors[:keep]

            overflow = [b for b in survivors[keep:] if b["_z"] >= bar]
            overflow = overflow[: max(0, int(self.lrs.overflow_cap * len(survivors))) ]

            for b in overflow:
                env: BranchEnvelope = b["_env"]
                leaves = b.get("leaves_cb", lambda h: [0.0])(1)
                env.update(self.compute_spent, leaves)

            survivors = keep_set + overflow

            for b in survivors:
                if b.get("_v", 0.0) >= self.mainline_bar + self.lrs.bar_delta:
                    env: BranchEnvelope = b["_env"]
                    leaves = b.get("leaves_cb", lambda h: [0.0])(self.horizons[horizon_idx])
                    env.update(self.compute_spent, leaves)
                    if env.history and env.history[-1][1] >= self.mainline_bar + self.lrs.bar_delta:
                        promoted = b
                        return survivors, promoted

            horizon_idx += 1

        return survivors, promoted

    def update_mainline_bar(self, v: float):
        if v > self.mainline_bar:
            delta = v - self.mainline_bar
            self.mainline_bar = v
            self.compute_spent += 1
            return self.plateau.update(self.compute_spent, delta)
        else:
            self.compute_spent += 1
            return self.plateau.update(self.compute_spent, 0.0)
