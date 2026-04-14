"""Event denoising pipeline (minimal version)."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .adapters import Event


@dataclass
class PreprocessStats:
    raw_count: int = 0
    after_refractory: int = 0
    after_isolated: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_count": self.raw_count,
            "after_refractory": self.after_refractory,
            "after_isolated": self.after_isolated,
        }


class EventPreprocessor:
    """Minimal preprocessing with refractory + isolated filtering."""

    def __init__(self, refractory_us: float, isolated_dt_us: float, isolated_radius: int) -> None:
        self.refractory_sec = refractory_us * 1e-6
        self.isolated_dt_sec = isolated_dt_us * 1e-6
        self.isolated_radius = max(0, isolated_radius)

    def filter_events(self, events: List[Event]) -> Tuple[List[Event], PreprocessStats]:
        stats = PreprocessStats(raw_count=len(events))
        stage1 = self.filter_refractory(events)
        stats.after_refractory = len(stage1)
        stage2 = self.filter_isolated(stage1)
        stats.after_isolated = len(stage2)
        return stage2, stats

    def filter_refractory(self, events: List[Event]) -> List[Event]:
        last_ts: Dict[Tuple[int, int, int], float] = {}
        out: List[Event] = []
        for e in events:
            key = (e.x, e.y, e.p)
            t_prev = last_ts.get(key)
            if t_prev is not None and (e.t - t_prev) < self.refractory_sec:
                continue
            last_ts[key] = e.t
            out.append(e)
        return out

    def filter_isolated(self, events: List[Event]) -> List[Event]:
        # Simple online support check in local spatio-temporal neighborhood.
        history: Dict[Tuple[int, int], float] = {}
        out: List[Event] = []
        r = self.isolated_radius
        for e in events:
            support = False
            for yy in range(e.y - r, e.y + r + 1):
                for xx in range(e.x - r, e.x + r + 1):
                    t = history.get((xx, yy))
                    if t is not None and abs(e.t - t) <= self.isolated_dt_sec:
                        support = True
                        break
                if support:
                    break
            history[(e.x, e.y)] = e.t
            if support:
                out.append(e)
        return out
