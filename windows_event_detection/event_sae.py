from __future__ import annotations

import numpy as np


class EventSAE:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> None:
        self.pos = np.full((self.height, self.width), -10**12, dtype=np.int64)
        self.neg = np.full((self.height, self.width), -10**12, dtype=np.int64)

    def update(self, events: np.ndarray) -> None:
        for e in events:
            x, y, t, p = int(e["x"]), int(e["y"]), int(e["t"]), int(e["p"])
            if 0 <= x < self.width and 0 <= y < self.height:
                if p > 0:
                    self.pos[y, x] = t
                else:
                    self.neg[y, x] = t

    def get_patch(self, x: int, y: int, radius: int, polarity: int) -> np.ndarray:
        x0, x1 = max(0, x - radius), min(self.width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(self.height - 1, y + radius)
        src = self.pos if polarity > 0 else self.neg
        return src[y0 : y1 + 1, x0 : x1 + 1]

    def get_timestamp(self, x: int, y: int, polarity: int) -> int:
        src = self.pos if polarity > 0 else self.neg
        return int(src[y, x])
