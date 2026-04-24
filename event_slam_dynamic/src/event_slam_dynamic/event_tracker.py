"""Short-term corner tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrackerConfig:
    match_radius_px: float = 8.0
    match_dt_us: int = 30000
    min_track_len: int = 4
    max_track_age: int = 6
    max_track_history: int = 15
    velocity_smooth_alpha: float = 0.6
    polarity_consistency: bool = False


@dataclass
class Track:
    track_id: int
    points: List[Dict[str, float]] = field(default_factory=list)
    hits: int = 0
    missed: int = 0
    age: int = 0
    confidence: float = 0.0
    residual: float = 0.0
    state: str = "uncertain"
    polarity: int = 1
    velocity: Tuple[float, float] = (0.0, 0.0)
    dyn_count: int = 0
    sta_count: int = 0


class EventTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.next_track_id = 1
        self.tracks: List[Track] = []

    def update(self, corners: Sequence[Dict[str, object]]) -> List[Track]:
        used = set()
        for tr in self.tracks:
            tr.age += 1
            tr.missed += 1

        for c in corners:
            idx = self._find_best_track(c, used)
            if idx is None:
                self._spawn_track(c)
            else:
                used.add(idx)
                self._append(self.tracks[idx], c)

        self.tracks = [t for t in self.tracks if t.missed <= self.cfg.max_track_age]
        return self.tracks

    def _find_best_track(self, c: Dict[str, object], used: set[int]) -> Optional[int]:
        cx, cy, ct = float(c["x"]), float(c["y"]), float(c["t"])
        best_i = None
        best_d = 1e9
        for i, tr in enumerate(self.tracks):
            if i in used or not tr.points:
                continue
            lp = tr.points[-1]
            dt = ct - float(lp["t"])
            if dt <= 0 or dt > self.cfg.match_dt_us:
                continue
            if self.cfg.polarity_consistency and int(c.get("polarity", 1)) != tr.polarity:
                continue
            pred_x = float(lp["x"]) + tr.velocity[0] * dt
            pred_y = float(lp["y"]) + tr.velocity[1] * dt
            d = float(np.hypot(cx - pred_x, cy - pred_y))
            if d < best_d and d <= self.cfg.match_radius_px:
                best_d, best_i = d, i
        return best_i

    def _append(self, tr: Track, c: Dict[str, object]) -> None:
        p = {"x": float(c["x"]), "y": float(c["y"]), "t": float(c["t"])}
        prev = tr.points[-1] if tr.points else None
        tr.points.append(p)
        tr.points = tr.points[-self.cfg.max_track_history :]
        tr.hits += 1
        tr.missed = 0
        tr.polarity = int(c.get("polarity", tr.polarity))
        if prev:
            dt = max(1.0, p["t"] - prev["t"])
            uv = ((p["x"] - prev["x"]) / dt, (p["y"] - prev["y"]) / dt)
            a = self.cfg.velocity_smooth_alpha
            tr.velocity = (a * uv[0] + (1 - a) * tr.velocity[0], a * uv[1] + (1 - a) * tr.velocity[1])
        tr.confidence = min(1.0, tr.hits / float(max(1, self.cfg.min_track_len)))

    def _spawn_track(self, c: Dict[str, object]) -> None:
        tr = Track(track_id=self.next_track_id)
        self.next_track_id += 1
        self._append(tr, c)
        self.tracks.append(tr)
