"""eFAST-like corner detector on SAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .event_sae import EventSAE


@dataclass
class CornerConfig:
    arc_small_radius: int = 3
    arc_large_radius: int = 4
    arc_min_length: int = 3
    arc_max_length: int = 10
    corner_score_threshold: float = 1500.0
    nms_radius: int = 2
    max_corners_per_window: int = 300
    grid_rows: int = 6
    grid_cols: int = 8
    max_corners_per_cell: int = 12


class EventCornerDetector:
    CIRCLE_3 = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
    CIRCLE_4 = [(0, -4), (1, -4), (2, -3), (3, -2), (4, -1), (4, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 4), (-1, 4), (-2, 3), (-3, 2), (-4, 1), (-4, 0), (-4, -1), (-3, -2), (-2, -3), (-1, -4)]

    def __init__(self, width: int, height: int, cfg: CornerConfig):
        self.width = width
        self.height = height
        self.cfg = cfg
        self.corner_id = 0

    def detect(self, events: Sequence[Dict[str, int]], sae: EventSAE) -> List[Dict[str, object]]:
        candidates = []
        for e in events:
            x, y, t, p = int(e["x"]), int(e["y"]), int(e["t"]), int(e.get("p", 1))
            score = max(self._arc_score(sae, x, y, t, p, self.CIRCLE_3), self._arc_score(sae, x, y, t, p, self.CIRCLE_4))
            sae.update(x, y, t, p)
            if score >= self.cfg.corner_score_threshold:
                self.corner_id += 1
                candidates.append({"corner_id": self.corner_id, "x": x, "y": y, "t": t, "polarity": p, "score": score})

        selected = self._nms(candidates)
        return self._grid_limit(selected)[: self.cfg.max_corners_per_window]

    def _arc_score(self, sae: EventSAE, x: int, y: int, t: int, p: int, circle: List[Tuple[int, int]]) -> float:
        ts = []
        for dx, dy in circle:
            ts.append(sae.get_time(x + dx, y + dy, p))
        if not ts:
            return 0.0
        age = [max(0, t - tt) for tt in ts]
        recent = [1 if a < 5000 else 0 for a in age]
        doubled = recent + recent
        max_run = run = 0
        for v in doubled:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
            if max_run >= len(recent):
                break
        if max_run < self.cfg.arc_min_length or max_run > self.cfg.arc_max_length:
            return 0.0
        return float(np.percentile(age, 70) - np.percentile(age, 20))

    def _nms(self, corners: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not corners:
            return []
        corners = sorted(corners, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        keep = []
        r2 = self.cfg.nms_radius * self.cfg.nms_radius
        for c in corners:
            x, y = int(c["x"]), int(c["y"])
            ok = True
            for k in keep:
                dx, dy = x - int(k["x"]), y - int(k["y"])
                if dx * dx + dy * dy <= r2:
                    ok = False
                    break
            if ok:
                keep.append(c)
        return keep

    def _grid_limit(self, corners: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not corners:
            return []
        cell_w = max(1, self.width // self.cfg.grid_cols)
        cell_h = max(1, self.height // self.cfg.grid_rows)
        buckets: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for c in corners:
            col = min(self.cfg.grid_cols - 1, int(c["x"]) // cell_w)
            row = min(self.cfg.grid_rows - 1, int(c["y"]) // cell_h)
            buckets.setdefault((row, col), []).append(c)
        out = []
        for _, vals in buckets.items():
            vals.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
            out.extend(vals[: self.cfg.max_corners_per_cell])
        out.sort(key=lambda c: int(c["t"]))
        return out
