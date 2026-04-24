"""Surface of Active Events (SAE) memory."""

from __future__ import annotations

import numpy as np


class EventSAE:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pos = np.full((height, width), -10**12, dtype=np.int64)
        self.neg = np.full((height, width), -10**12, dtype=np.int64)

    def update(self, x: int, y: int, t: int, p: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            (self.pos if p > 0 else self.neg)[y, x] = t

    def get_time(self, x: int, y: int, p: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -10**12
        return int((self.pos if p > 0 else self.neg)[y, x])

    def get_patch(self, x: int, y: int, radius: int, p: int):
        x0, x1 = max(0, x - radius), min(self.width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(self.height - 1, y + radius)
        return (self.pos if p > 0 else self.neg)[y0 : y1 + 1, x0 : x1 + 1]
