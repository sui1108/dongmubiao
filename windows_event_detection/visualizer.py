from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from config import DetectionConfig
from event_types import DynamicObject, Track


class DetectionVisualizer:
    def __init__(self, config: DetectionConfig, width: int, height: int, frame_dir: Path) -> None:
        self.cfg = config
        self.width = width
        self.height = height
        self.frame_dir = frame_dir
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def _event_image(self, events: np.ndarray) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if len(events) == 0:
            return img
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)
        p = events["p"].astype(np.int32)
        img[y[p > 0], x[p > 0], 1] = 255
        img[y[p <= 0], x[p <= 0], 2] = 255
        return img

    def draw_and_save(
        self,
        frame_idx: int,
        events: np.ndarray,
        static_tracks: List[Track],
        dynamic_tracks: List[Track],
        uncertain_tracks: List[Track],
        objects: List[DynamicObject],
        stats: Dict[str, float],
    ) -> Path:
        canvas = self._event_image(events)

        for t in static_tracks:
            if t.points:
                cv2.circle(canvas, (int(t.points[-1].x), int(t.points[-1].y)), 2, (255, 255, 0), -1)
        for t in dynamic_tracks:
            if t.points:
                cv2.circle(canvas, (int(t.points[-1].x), int(t.points[-1].y)), 3, (0, 255, 0), -1)
        for t in uncertain_tracks:
            if t.points:
                cv2.circle(canvas, (int(t.points[-1].x), int(t.points[-1].y)), 2, (100, 100, 255), -1)

        for obj in objects:
            color = (0, 255, 255) if obj.predicted else (0, 165, 255)
            x0, y0, x1, y1 = [int(v) for v in obj.last_bbox]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            state = "predicted" if obj.predicted else "detected"
            label = f"Obj-ID:{obj.object_id} tc:{obj.track_count} mcs:{obj.motion_consistency_score:.2f} {state}"
            cv2.putText(canvas, label, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        text_lines = [
            f"frame:{frame_idx}",
            f"raw_events:{int(stats['raw_count'])} filtered:{int(stats['filtered_count'])}",
            f"corners:{int(stats['corner_count'])} active_tracks:{int(stats['active_track_count'])}",
            f"dynamic_tracks:{int(stats['dynamic_track_count'])} dynamic_objects:{int(stats['dynamic_object_count'])}",
            f"processing_ms:{stats['processing_ms']:.2f}",
        ]
        yy = 18
        for line in text_lines:
            cv2.putText(canvas, line, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            yy += 18

        out = self.frame_dir / f"frame_{frame_idx:06d}.png"
        ok = cv2.imwrite(str(out), canvas)
        if not ok:
            raise RuntimeError(f"Failed to save visualization image: {out}")

        if self.cfg.show_window:
            cv2.imshow("event_detection", canvas)
            cv2.waitKey(1)

        return out
