"""Track motion classifier based on background flow residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .event_tracker import Track


@dataclass
class MotionClassifierConfig:
    dynamic_residual_threshold: float = 0.00055
    static_residual_threshold: float = 0.00025
    dynamic_confirm_count: int = 2
    static_confirm_count: int = 2
    fallback_enable: bool = True
    fallback_dynamic_ratio_min: float = 0.05
    adaptive_threshold_enable: bool = True


class MotionClassifier:
    def __init__(self, cfg: MotionClassifierConfig):
        self.cfg = cfg
        self.zero_dynamic_streak = 0

    def classify(self, tracks: List[Track], min_track_len: int) -> Tuple[List[Track], Dict[str, float]]:
        stable = [t for t in tracks if len(t.points) >= min_track_len]
        if stable:
            flows = np.array([[t.velocity[0], t.velocity[1]] for t in stable], dtype=float)
            bg = np.median(flows, axis=0)
        else:
            bg = np.array([0.0, 0.0], dtype=float)

        residuals = []
        dyn_cnt = 0
        threshold = self.cfg.dynamic_residual_threshold
        if self.cfg.adaptive_threshold_enable and self.zero_dynamic_streak >= 5 and stable:
            threshold *= 0.8

        for t in tracks:
            vel = np.array(t.velocity)
            r = float(np.linalg.norm(vel - bg))
            t.residual = r
            residuals.append(r)
            if len(t.points) < min_track_len:
                t.state = "uncertain"
                continue
            if r >= threshold:
                t.dyn_count += 1
                t.sta_count = max(0, t.sta_count - 1)
            elif r <= self.cfg.static_residual_threshold:
                t.sta_count += 1
                t.dyn_count = max(0, t.dyn_count - 1)
            if t.dyn_count >= self.cfg.dynamic_confirm_count:
                t.state = "dynamic"
            elif t.sta_count >= self.cfg.static_confirm_count:
                t.state = "static"
            else:
                t.state = "uncertain"
            dyn_cnt += 1 if t.state == "dynamic" else 0

        fallback_triggered = False
        if self.cfg.fallback_enable and stable:
            ratio = dyn_cnt / float(max(1, len(stable)))
            if ratio < self.cfg.fallback_dynamic_ratio_min and np.mean(residuals) > self.cfg.static_residual_threshold * 1.2:
                fallback_triggered = True
                high = np.percentile(residuals, 75) if residuals else threshold
                for t in tracks:
                    if len(t.points) >= min_track_len and t.residual >= high:
                        t.state = "dynamic"

        dyn_after = len([t for t in tracks if t.state == "dynamic"])
        self.zero_dynamic_streak = self.zero_dynamic_streak + 1 if dyn_after == 0 and len(stable) > 0 else 0
        stats = {
            "avg_residual": float(np.mean(residuals)) if residuals else 0.0,
            "max_residual": float(np.max(residuals)) if residuals else 0.0,
            "bg_u": float(bg[0]),
            "bg_v": float(bg[1]),
            "fallback_triggered": fallback_triggered,
            "dynamic_threshold_used": threshold,
        }
        return tracks, stats
