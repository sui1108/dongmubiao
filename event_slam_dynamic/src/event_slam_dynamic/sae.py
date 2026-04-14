"""Surface of Active Events (SAE) / Time Surface manager."""

from typing import List, Optional

import numpy as np

from .adapters import Event


class TimeSurface:
    """Maintain latest timestamp per pixel (optionally per polarity)."""

    def __init__(self, width: int, height: int, separate_polarity: bool = True) -> None:
        self.width = width
        self.height = height
        self.separate_polarity = separate_polarity
        self.latest_time: float = 0.0
        self.reset()

    def reset(self) -> None:
        self._ts_pos = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        self._ts_neg = np.full((self.height, self.width), -np.inf, dtype=np.float32)

    def update(self, events: List[Event]) -> None:
        for ev in events:
            self.update_one(ev)

    def update_one(self, event: Event) -> None:
        if not (0 <= event.x < self.width and 0 <= event.y < self.height):
            return
        if self.separate_polarity:
            if event.p > 0:
                self._ts_pos[event.y, event.x] = event.t
            else:
                self._ts_neg[event.y, event.x] = event.t
        else:
            self._ts_pos[event.y, event.x] = event.t
        self.latest_time = max(self.latest_time, event.t)

    def get_patch(self, x: int, y: int, radius: int, polarity: Optional[int] = None) -> np.ndarray:
        x0, x1 = max(0, x - radius), min(self.width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.height, y + radius + 1)

        if not self.separate_polarity:
            return self._ts_pos[y0:y1, x0:x1]
        if polarity is None:
            return np.maximum(self._ts_pos[y0:y1, x0:x1], self._ts_neg[y0:y1, x0:x1])
        return self._ts_pos[y0:y1, x0:x1] if polarity > 0 else self._ts_neg[y0:y1, x0:x1]

    def to_decay_image(self, tau_sec: float = 0.03) -> np.ndarray:
        """Return normalized [0,1] decay image for debug visualization."""
        merged = np.maximum(self._ts_pos, self._ts_neg)
        age = np.maximum(0.0, self.latest_time - merged)
        img = np.exp(-age / max(1e-6, tau_sec))
        img[np.isinf(merged)] = 0.0
        return img.astype(np.float32)
