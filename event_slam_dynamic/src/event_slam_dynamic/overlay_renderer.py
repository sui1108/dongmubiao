"""OpenCV overlay helpers for track/cluster rendering."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def draw_tracks(
    image: np.ndarray,
    tracks: Sequence[Dict[str, object]],
    color: Tuple[int, int, int],
    max_history: int = 20,
    show_track_id: bool = False,
) -> None:
    """Draw polyline tracks on a BGR image."""
    for track in tracks:
        points = track.get("points", [])[-max_history:]
        if len(points) < 1:
            continue
        pts = np.array([[int(p.get("x", 0)), int(p.get("y", 0))] for p in points], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(image, tuple(pts[-1]), 2, color, -1)
        if show_track_id and "track_id" in track:
            cv2.putText(
                image,
                f"id:{track['track_id']}",
                (int(pts[-1][0]) + 3, int(pts[-1][1]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def draw_clusters(image: np.ndarray, clusters: Iterable[Dict[str, object]], color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw cluster bbox and metadata."""
    for cluster in clusters:
        bbox = cluster.get("bbox", {})
        x_min = int(bbox.get("x_min", 0))
        y_min = int(bbox.get("y_min", 0))
        x_max = int(bbox.get("x_max", 0))
        y_max = int(bbox.get("y_max", 0))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
        center = cluster.get("center", {})
        cx = int(center.get("x", (x_min + x_max) / 2.0))
        cy = int(center.get("y", (y_min + y_max) / 2.0))
        cv2.circle(image, (cx, cy), 2, color, -1)
        mv = cluster.get("mean_velocity", {})
        speed = (float(mv.get("u", 0.0)) ** 2 + float(mv.get("v", 0.0)) ** 2) ** 0.5
        label = f"c{cluster.get('cluster_id', -1)} n={cluster.get('track_count', 0)} v={speed:.2f}"
        cv2.putText(image, label, (x_min, max(12, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_info_block(image: np.ndarray, lines: List[str], origin: Tuple[int, int] = (8, 18)) -> None:
    """Draw semi-transparent text panel."""
    x0, y0 = origin
    line_h = 16
    panel_h = line_h * (len(lines) + 1)
    panel_w = max(220, max(len(s) for s in lines) * 7 if lines else 220)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 14), (x0 + panel_w, y0 - 14 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    for idx, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (x0, y0 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
