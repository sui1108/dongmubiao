from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from config import DetectionConfig


class EventDenoiser:
    def __init__(self, config: DetectionConfig, width: int, height: int) -> None:
        self.cfg = config
        self.width = width
        self.height = height
        self.last_ts = np.full((height, width), -10**12, dtype=np.int64)

    def _apply_refractory(self, events: np.ndarray) -> np.ndarray:
        keep = np.zeros(len(events), dtype=bool)
        for i, e in enumerate(events):
            x, y, t = int(e["x"]), int(e["y"]), int(e["t"])
            if t - self.last_ts[y, x] >= self.cfg.refractory_us:
                keep[i] = True
                self.last_ts[y, x] = t
        return events[keep]

    def _apply_support_filter(self, events: np.ndarray) -> np.ndarray:
        if len(events) == 0:
            return events
        last_local = np.full((self.height, self.width), -10**12, dtype=np.int64)
        keep = np.zeros(len(events), dtype=bool)
        r = self.cfg.support_radius
        for i, e in enumerate(events):
            x, y, t = int(e["x"]), int(e["y"]), int(e["t"])
            x0, x1 = max(0, x - r), min(self.width - 1, x + r)
            y0, y1 = max(0, y - r), min(self.height - 1, y + r)
            patch = last_local[y0 : y1 + 1, x0 : x1 + 1]
            support = np.count_nonzero((t - patch) <= self.cfg.support_dt_us)
            if support >= self.cfg.min_support_count:
                keep[i] = True
            last_local[y, x] = t
        return events[keep]

    def _apply_hot_pixel(self, events: np.ndarray) -> np.ndarray:
        if len(events) == 0:
            return events
        key = events["y"].astype(np.int64) * self.width + events["x"].astype(np.int64)
        uniq, cnt = np.unique(key, return_counts=True)
        hot = set(uniq[cnt > self.cfg.hot_pixel_count_threshold].tolist())
        if not hot:
            return events
        mask = np.array([k not in hot for k in key], dtype=bool)
        return events[mask]

    def filter(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        raw_count = len(events)
        if raw_count == 0:
            return events, {
                "raw_count": 0,
                "after_refractory_count": 0,
                "after_support_count": 0,
                "after_hot_pixel_count": 0,
                "final_count": 0,
                "removed_ratio": 0.0,
                "fallback_relaxed_filter_used": False,
            }

        ev1 = self._apply_refractory(events)
        ev2 = self._apply_support_filter(ev1)
        ev3 = self._apply_hot_pixel(ev2)
        final = ev3
        fallback = False

        if len(final) / max(raw_count, 1) < self.cfg.min_event_keep_ratio:
            fallback = True
            final = ev1 if len(ev1) > 0 else events

        stats = {
            "raw_count": int(raw_count),
            "after_refractory_count": int(len(ev1)),
            "after_support_count": int(len(ev2)),
            "after_hot_pixel_count": int(len(ev3)),
            "final_count": int(len(final)),
            "removed_ratio": float(1.0 - len(final) / max(raw_count, 1)),
            "fallback_relaxed_filter_used": fallback,
        }
        return final, stats
