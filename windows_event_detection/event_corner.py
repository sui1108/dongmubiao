from __future__ import annotations

from typing import List, Tuple
import numpy as np

from config import DetectionConfig
from event_sae import EventSAE
from event_types import Corner


CIRCLE_R3 = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)]
CIRCLE_R4 = [(-4, 0), (-4, 1), (-3, 3), (-1, 4), (0, 4), (1, 4), (3, 3), (4, 1), (4, 0), (4, -1), (3, -3), (1, -4), (0, -4), (-1, -4), (-3, -3), (-4, -1)]


class EventCornerDetector:
    def __init__(self, config: DetectionConfig, sae: EventSAE, width: int, height: int) -> None:
        self.cfg = config
        self.sae = sae
        self.width = width
        self.height = height
        self.corner_id = 0

    def _recent_arc_score(self, x: int, y: int, t: int, p: int, circle: List[Tuple[int, int]]) -> float:
        vals = []
        for dx, dy in circle:
            xx, yy = x + dx, y + dy
            if xx < 0 or yy < 0 or xx >= self.width or yy >= self.height:
                vals.append(False)
                continue
            recent = (t - self.sae.get_timestamp(xx, yy, p)) <= self.cfg.sae_recent_us
            vals.append(bool(recent))
        vals2 = vals + vals
        max_run = 0
        run = 0
        for v in vals2:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
            if run >= len(vals):
                break
        if self.cfg.arc_min_length <= max_run <= self.cfg.arc_max_length:
            return float(max_run / len(vals))
        return 0.0

    def detect(self, events: np.ndarray) -> List[Corner]:
        candidates: List[Corner] = []
        circle_small = CIRCLE_R3 if self.cfg.arc_small_radius == 3 else CIRCLE_R4
        circle_large = CIRCLE_R4
        for e in events:
            x, y, t, p = int(e["x"]), int(e["y"]), int(e["t"]), int(e["p"])
            s1 = self._recent_arc_score(x, y, t, p, circle_small)
            s2 = self._recent_arc_score(x, y, t, p, circle_large)
            score = max(s1, s2)
            if score >= self.cfg.corner_score_threshold:
                self.corner_id += 1
                candidates.append(Corner(self.corner_id, x, y, t, p, score))

        candidates.sort(key=lambda c: c.score, reverse=True)
        kept: List[Corner] = []
        cell_h = max(1, self.height // self.cfg.grid_rows)
        cell_w = max(1, self.width // self.cfg.grid_cols)
        cell_count = {}
        for c in candidates:
            if len(kept) >= self.cfg.max_corners_per_window:
                break
            cell = (int(c.y // cell_h), int(c.x // cell_w))
            if cell_count.get(cell, 0) >= self.cfg.max_corners_per_cell:
                continue
            ok = True
            for k in kept:
                if (k.x - c.x) ** 2 + (k.y - c.y) ** 2 <= self.cfg.nms_radius ** 2:
                    ok = False
                    break
            if ok:
                kept.append(c)
                cell_count[cell] = cell_count.get(cell, 0) + 1
        return kept
