"""Event denoising utilities for dynamic front-end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EventDenoiseConfig:
    refractory_us: int = 80
    support_radius: int = 1
    support_dt_us: int = 3000
    min_support_count: int = 1
    hot_pixel_count_threshold: int = 120
    min_event_keep_ratio: float = 0.12


class EventDenoiser:
    """Refractory + support + hot-pixel denoising with fallback protection."""

    def __init__(self, width: int, height: int, cfg: EventDenoiseConfig):
        self.width = width
        self.height = height
        self.cfg = cfg
        self.last_ts = np.full((height, width), -10**12, dtype=np.int64)

    def filter(self, events: List[Dict[str, int]]) -> Tuple[List[Dict[str, int]], Dict[str, object]]:
        raw_count = len(events)
        if raw_count == 0:
            return [], {
                "raw_count": 0,
                "after_refractory_count": 0,
                "after_support_count": 0,
                "after_hot_pixel_count": 0,
                "final_count": 0,
                "removed_ratio": 0.0,
                "fallback_relaxed_filter_used": False,
            }

        refractory_events = self._refractory(events)
        support_events = self._support_filter(refractory_events, self.cfg.min_support_count)
        final_events = self._hot_pixel_filter(support_events)
        fallback = False
        if len(final_events) < max(1, int(raw_count * self.cfg.min_event_keep_ratio)):
            fallback = True
            relaxed_support = max(0, self.cfg.min_support_count - 1)
            support_events = self._support_filter(refractory_events, relaxed_support)
            final_events = self._hot_pixel_filter(support_events, relaxed=True)

        stats = {
            "raw_count": raw_count,
            "after_refractory_count": len(refractory_events),
            "after_support_count": len(support_events),
            "after_hot_pixel_count": len(final_events),
            "final_count": len(final_events),
            "removed_ratio": float(max(0.0, 1.0 - len(final_events) / float(raw_count))),
            "fallback_relaxed_filter_used": fallback,
        }
        return final_events, stats

    def _refractory(self, events: List[Dict[str, int]]) -> List[Dict[str, int]]:
        out = []
        for e in events:
            x, y, t = int(e["x"]), int(e["y"]), int(e["t"])
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            if t - int(self.last_ts[y, x]) >= self.cfg.refractory_us:
                out.append(e)
                self.last_ts[y, x] = t
        return out

    def _support_filter(self, events: List[Dict[str, int]], min_support: int) -> List[Dict[str, int]]:
        if min_support <= 0:
            return list(events)
        latest = np.full((self.height, self.width), -10**12, dtype=np.int64)
        out: List[Dict[str, int]] = []
        r = self.cfg.support_radius
        dt = self.cfg.support_dt_us
        for e in events:
            x, y, t = int(e["x"]), int(e["y"]), int(e["t"])
            x0, x1 = max(0, x - r), min(self.width - 1, x + r)
            y0, y1 = max(0, y - r), min(self.height - 1, y + r)
            patch = latest[y0 : y1 + 1, x0 : x1 + 1]
            support = int(np.count_nonzero((t - patch) <= dt))
            if support >= min_support:
                out.append(e)
            latest[y, x] = t
        return out

    def _hot_pixel_filter(self, events: List[Dict[str, int]], relaxed: bool = False) -> List[Dict[str, int]]:
        if not events:
            return []
        threshold = self.cfg.hot_pixel_count_threshold * (2 if relaxed else 1)
        counts: Dict[Tuple[int, int], int] = {}
        for e in events:
            key = (int(e["x"]), int(e["y"]))
            counts[key] = counts.get(key, 0) + 1
        hot = {k for k, v in counts.items() if v > threshold}
        if not hot:
            return list(events)
        return [e for e in events if (int(e["x"]), int(e["y"])) not in hot]
