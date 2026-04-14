"""Corner detector abstraction + minimal heuristic implementation.

TODO: replace heuristic with FA-Harris / eHarris / Arc* style detectors.
"""

from typing import List

import numpy as np

from .adapters import Corner, Event
from .sae import TimeSurface


class CornerDetector:
    """Heuristic corner detector operating on local time-surface patch contrast."""

    def __init__(self, max_corners: int, score_threshold: float) -> None:
        self.max_corners = max_corners
        self.score_threshold = score_threshold
        self._next_id = 1

    def detect(self, events: List[Event], sae: TimeSurface) -> List[Corner]:
        corners: List[Corner] = []
        for e in events:
            patch = sae.get_patch(e.x, e.y, radius=1, polarity=e.p)
            if patch.size < 9:
                continue
            center = patch[1, 1]
            if not np.isfinite(center):
                continue
            neigh = np.delete(patch.reshape(-1), 4)
            neigh = neigh[np.isfinite(neigh)]
            if neigh.size < 3:
                continue
            score = float(np.mean(np.abs(center - neigh)))
            if score < self.score_threshold:
                continue
            corners.append(
                Corner(
                    id=self._next_id,
                    x=float(e.x),
                    y=float(e.y),
                    t=e.t,
                    score=score,
                    polarity=e.p,
                )
            )
            self._next_id += 1
        corners.sort(key=lambda c: c.score, reverse=True)
        return corners[: self.max_corners]
