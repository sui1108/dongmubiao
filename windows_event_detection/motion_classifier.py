from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from config import DetectionConfig
from event_types import Track


class MotionClassifier:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.no_dynamic_streak = 0

    def classify(self, tracks: List[Track], event_count: int, corner_count: int) -> Tuple[List[Track], List[Track], List[Track], Dict[str, float]]:
        stable = [t for t in tracks if len(t.points) >= self.cfg.min_track_len]
        if stable:
            bg_v = np.median(np.array([t.velocity for t in stable], dtype=np.float32), axis=0)
        else:
            bg_v = np.array([0.0, 0.0], dtype=np.float32)

        dyn_thr = self.cfg.dynamic_residual_threshold
        fallback_triggered = False
        if self.cfg.adaptive_threshold_enable and self.no_dynamic_streak >= 8:
            dyn_thr *= 0.8

        static_tracks: List[Track] = []
        dynamic_tracks: List[Track] = []
        uncertain_tracks: List[Track] = []

        for t in tracks:
            obs = np.array(t.velocity, dtype=np.float32)
            residual = float(np.linalg.norm(obs - bg_v))
            t.residual = residual
            if residual >= dyn_thr:
                t.dyn_count += 1
                t.sta_count = 0
            elif residual <= self.cfg.static_residual_threshold:
                t.sta_count += 1
                t.dyn_count = 0
            if t.dyn_count >= self.cfg.dynamic_confirm_count:
                t.state = "dynamic"
                dynamic_tracks.append(t)
            elif t.sta_count >= self.cfg.static_confirm_count:
                t.state = "static"
                static_tracks.append(t)
            else:
                t.state = "uncertain"
                uncertain_tracks.append(t)

        if not dynamic_tracks:
            self.no_dynamic_streak += 1
        else:
            self.no_dynamic_streak = 0

        if self.cfg.fallback_enable and self.no_dynamic_streak >= 5 and event_count > 500 and corner_count > 40:
            fallback_triggered = True
            for t in uncertain_tracks:
                if t.residual >= dyn_thr * 0.85:
                    t.state = "dynamic"
                    dynamic_tracks.append(t)
            uncertain_tracks = [t for t in uncertain_tracks if t.state != "dynamic"]

        return static_tracks, dynamic_tracks, uncertain_tracks, {
            "bg_vx": float(bg_v[0]),
            "bg_vy": float(bg_v[1]),
            "fallback_triggered": fallback_triggered,
            "dynamic_threshold": dyn_thr,
        }
