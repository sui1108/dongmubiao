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

    def draw_and_save(self, frame_idx: int, events: np.ndarray, static_tracks: List[Track], dynamic_tracks: List[Track], uncertain_tracks: List[Track], objects: Dict[str, List[DynamicObject]], stats: Dict[str, float]) -> Path:
        canvas = self._event_image(events)
        primary = objects.get('primary_objects', [])
        raw = objects.get('raw_clusters', [])
        tracked = objects.get('tracked_objects', [])
        display_objects = primary if self.cfg.show_final_objects_only else tracked

        if self.cfg.show_raw_clusters:
            for obj in raw:
                x0,y0,x1,y1=[int(v) for v in obj.last_bbox]
                cv2.rectangle(canvas,(x0,y0),(x1,y1),(130,130,130),1,cv2.LINE_AA)

        for obj in display_objects:
            if obj.state == 'tentative' and not self.cfg.show_tentative_objects:
                continue
            if obj.state == 'lost' and not self.cfg.show_lost_objects:
                continue
            if obj.suppressed_by and not self.cfg.show_child_boxes:
                continue
            color = (180, 180, 90) if obj.state == 'lost' else ((120, 120, 120) if obj.state == 'tentative' else (0, 165, 255))
            x0, y0, x1, y1 = [int(v) for v in obj.last_bbox]
            line_t = cv2.LINE_4 if obj.state == 'lost' else cv2.LINE_AA
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, line_t)
            text = f"Obj-ID:{obj.object_id} {obj.state} age:{obj.age} hits:{obj.hits} d:{obj.density:.3f} reused_id:{obj.reused_id} reverse_motion:{obj.reverse_motion}"
            cv2.putText(canvas, text, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        cv2.putText(canvas, f"frame:{frame_idx} primary:{len(primary)} tracked:{len(tracked)} raw:{len(raw)}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out = self.frame_dir / f"frame_{frame_idx:06d}.png"
        if self.cfg.visualization_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.cfg.visualization_scale, fy=self.cfg.visualization_scale)
        ok = cv2.imwrite(str(out), canvas)
        if not ok:
            raise RuntimeError(f"Failed to save visualization image: {out}")
        return out
