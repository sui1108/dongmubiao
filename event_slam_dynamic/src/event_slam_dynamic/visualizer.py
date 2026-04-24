"""Canvas preparation and composed render pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from .overlay_renderer import draw_clusters, draw_info_block, draw_tracks


class VisualizationComposer:
    """Builds output frames from tracks, clusters and optional event background."""

    def __init__(self, width: int, height: int, show_track_ids: bool = True) -> None:
        self.width = width
        self.height = height
        self.show_track_ids = show_track_ids

    def make_background(self, events: Sequence[Dict[str, object]] | None = None, use_event_accum: bool = True) -> np.ndarray:
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not use_event_accum or not events:
            return bg

        for e in events:
            x = int(e.get("x", -1))
            y = int(e.get("y", -1))
            p = int(e.get("p", 1))
            if 0 <= x < self.width and 0 <= y < self.height:
                bg[y, x] = (255, 120, 60) if p > 0 else (60, 120, 255)

        return cv2.GaussianBlur(bg, (3, 3), 0)

    def compose(
        self,
        events: Sequence[Dict[str, object]] | None,
        static_tracks: Sequence[Dict[str, object]],
        dynamic_tracks: Sequence[Dict[str, object]],
        uncertain_tracks: Sequence[Dict[str, object]],
        clusters: Iterable[Dict[str, object]],
        info_lines: List[str],
        use_event_accum: bool = True,
    ) -> np.ndarray:
        image = self.make_background(events, use_event_accum=use_event_accum)
        draw_tracks(image, static_tracks, (0, 255, 0), show_track_id=False)
        draw_tracks(image, dynamic_tracks, (0, 0, 255), show_track_id=self.show_track_ids)
        draw_tracks(image, uncertain_tracks, (0, 255, 255), show_track_id=False)
        draw_clusters(image, clusters, color=(0, 0, 255))
        draw_info_block(image, info_lines)
        return image
