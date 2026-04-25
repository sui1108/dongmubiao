from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np

from config import DetectionConfig
from event_types import DynamicObject, Track


def _cos_sim(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    a = math.hypot(v1[0], v1[1])
    b = math.hypot(v2[0], v2[1])
    if a < 1e-6 or b < 1e-6:
        return 1.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (a * b)


class DynamicObjectClusterer:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.next_obj_id = 1
        self.objects: Dict[int, DynamicObject] = {}

    def _build_clusters(self, tracks: List[Track]) -> List[List[Track]]:
        clusters: List[List[Track]] = []
        for t in tracks:
            pt = t.points[-1] if t.points else None
            if pt is None:
                continue
            assigned = False
            for c in clusters:
                cpt = c[0].points[-1]
                d = math.hypot(pt.x - cpt.x, pt.y - cpt.y)
                if d <= self.cfg.cluster_spatial_threshold and _cos_sim(t.velocity, c[0].velocity) >= self.cfg.cluster_velocity_cos_threshold:
                    c.append(t)
                    assigned = True
                    break
            if not assigned:
                clusters.append([t])
        return [c for c in clusters if len(c) >= self.cfg.cluster_min_tracks]

    def _cluster_to_bbox(self, cluster: List[Track]) -> Tuple[int, int, int, int]:
        xs = [int(t.points[-1].x) for t in cluster if t.points]
        ys = [int(t.points[-1].y) for t in cluster if t.points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return (max(0, x0 - 8), max(0, y0 - 8), x1 + 8, y1 + 8)

    def update(self, dynamic_tracks: List[Track], uncertain_tracks: List[Track], events: np.ndarray) -> Tuple[List[DynamicObject], int]:
        candidates = list(dynamic_tracks)
        if len(candidates) < self.cfg.cluster_min_tracks:
            candidates += [t for t in uncertain_tracks if t.residual >= 1.2 * self.cfg.static_residual_threshold]
        clusters = self._build_clusters(candidates)

        detected_objs: List[DynamicObject] = []
        for cluster in clusters:
            bbox = self._cluster_to_bbox(cluster)
            cx = float(np.mean([t.points[-1].x for t in cluster]))
            cy = float(np.mean([t.points[-1].y for t in cluster]))
            vv = np.mean(np.array([t.velocity for t in cluster], dtype=np.float32), axis=0)
            conf = float(np.clip(np.mean([t.confidence for t in cluster]), 0.2, 1.0))
            detected_objs.append(DynamicObject(0, bbox, (cx, cy), (float(vv[0]), float(vv[1])), 0, conf, False, [t.track_id for t in cluster]))

        predicted_count = 0
        used = set()
        for obj in detected_objs:
            best_id, best_dist = -1, 1e9
            for oid, prev in self.objects.items():
                d = math.hypot(obj.last_center[0] - prev.last_center[0], obj.last_center[1] - prev.last_center[1])
                if d < best_dist and d < self.cfg.cluster_spatial_threshold * 1.5:
                    best_dist, best_id = d, oid
            if best_id >= 0:
                obj.object_id = best_id
                used.add(best_id)
            else:
                obj.object_id = self.next_obj_id
                self.next_obj_id += 1
            self.objects[obj.object_id] = obj

        # keep predicted objects
        for oid, prev in list(self.objects.items()):
            if oid in used or any(o.object_id == oid for o in detected_objs):
                continue
            prev.missed_count += 1
            if prev.missed_count <= self.cfg.object_lost_tolerance_frames:
                px = int(prev.last_center[0] + prev.velocity[0] * 0.01)
                py = int(prev.last_center[1] + prev.velocity[1] * 0.01)
                w = prev.last_bbox[2] - prev.last_bbox[0]
                h = prev.last_bbox[3] - prev.last_bbox[1]
                pred_bbox = (px - w // 2, py - h // 2, px + w // 2, py + h // 2)
                support = 0
                if len(events):
                    ex = events["x"].astype(np.int32)
                    ey = events["y"].astype(np.int32)
                    support = int(np.count_nonzero((ex >= pred_bbox[0]) & (ex <= pred_bbox[2]) & (ey >= pred_bbox[1]) & (ey <= pred_bbox[3])))
                if support >= self.cfg.object_min_event_support:
                    prev.last_bbox = pred_bbox
                    prev.last_center = (px, py)
                    prev.predicted = True
                    detected_objs.append(prev)
                    predicted_count += 1
                else:
                    del self.objects[oid]
            else:
                del self.objects[oid]

        return detected_objs, predicted_count
