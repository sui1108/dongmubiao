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
        self.last_child_count: int = 0

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    @staticmethod
    def _bbox_edge_distance(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        dx = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
        dy = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
        return math.hypot(dx, dy)

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

        child_objs: List[DynamicObject] = []
        for cluster in clusters:
            bbox = self._cluster_to_bbox(cluster)
            cx = float(np.mean([t.points[-1].x for t in cluster]))
            cy = float(np.mean([t.points[-1].y for t in cluster]))
            vv = np.mean(np.array([t.velocity for t in cluster], dtype=np.float32), axis=0)
            conf = float(np.clip(np.mean([t.confidence for t in cluster]), 0.2, 1.0))
            child_objs.append(DynamicObject(0, bbox, (cx, cy), (float(vv[0]), float(vv[1])), 0, conf, False, [t.track_id for t in cluster]))

        self.last_child_count = len(child_objs)
        detected_objs = self._merge_primary_objects(child_objs, events)

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

    def _merge_primary_objects(self, child_objs: List[DynamicObject], events: np.ndarray) -> List[DynamicObject]:
        if not child_objs:
            return []
        n = len(child_objs)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(child_objs[i], child_objs[j], events):
                    union(i, j)

        groups: Dict[int, List[DynamicObject]] = {}
        for i, obj in enumerate(child_objs):
            groups.setdefault(find(i), []).append(obj)

        return [self._build_primary_object(parts) for parts in groups.values()]

    def _should_merge(self, o1: DynamicObject, o2: DynamicObject, events: np.ndarray) -> bool:
        edge_d = self._bbox_edge_distance(o1.last_bbox, o2.last_bbox)
        if edge_d >= self.cfg.primary_edge_distance_threshold:
            return False
        c1, c2 = self._bbox_center(o1.last_bbox), self._bbox_center(o2.last_bbox)
        if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) >= self.cfg.primary_center_distance_threshold:
            return False
        connected = self._has_event_bridge(o1.last_bbox, o2.last_bbox, events)
        same_history = self._near_same_history(o1.last_center, o2.last_center)
        return connected or same_history

    def _has_event_bridge(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], events: np.ndarray) -> bool:
        if len(events) == 0:
            return False
        x0 = max(0, min(b1[0], b2[0]))
        y0 = max(0, min(b1[1], b2[1]))
        x1 = max(b1[2], b2[2])
        y1 = max(b1[3], b2[3])
        ex = events["x"].astype(np.int32)
        ey = events["y"].astype(np.int32)
        support = int(np.count_nonzero((ex >= x0) & (ex <= x1) & (ey >= y0) & (ey <= y1)))
        return support >= self.cfg.object_min_event_support

    def _near_same_history(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> bool:
        threshold = self.cfg.primary_center_distance_threshold
        near1 = any(math.hypot(c1[0] - o.last_center[0], c1[1] - o.last_center[1]) < threshold for o in self.objects.values())
        near2 = any(math.hypot(c2[0] - o.last_center[0], c2[1] - o.last_center[1]) < threshold for o in self.objects.values())
        return near1 and near2

    def _build_primary_object(self, parts: List[DynamicObject]) -> DynamicObject:
        x0 = min(o.last_bbox[0] for o in parts)
        y0 = min(o.last_bbox[1] for o in parts)
        x1 = max(o.last_bbox[2] for o in parts)
        y1 = max(o.last_bbox[3] for o in parts)
        w, h = x1 - x0, y1 - y0
        pad_x = int(w * self.cfg.primary_padding_ratio)
        pad_y = int(h * self.cfg.primary_padding_ratio)
        bbox = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
        bbox = self._clamp_primary_bbox(bbox)
        center = self._bbox_center(bbox)
        vv = np.mean(np.array([o.velocity for o in parts], dtype=np.float32), axis=0)
        conf = float(np.clip(np.mean([o.confidence for o in parts]), 0.2, 1.0))
        track_ids: List[int] = []
        for o in parts:
            track_ids.extend(o.track_ids)
        return DynamicObject(0, bbox, center, (float(vv[0]), float(vv[1])), 0, conf, False, track_ids)

    def _clamp_primary_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        max_w = int(self.cfg.width * self.cfg.max_primary_bbox_width_ratio)
        max_h = int(self.cfg.height * self.cfg.max_primary_bbox_height_ratio)
        max_a = int(self.cfg.width * self.cfg.height * self.cfg.max_primary_bbox_area_ratio)
        scale = min(1.0, max_w / w, max_h / h, math.sqrt(max_a / float(w * h)))
        if scale < 1.0:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            x0 = int(cx - nw / 2)
            x1 = x0 + nw
            y0 = int(cy - nh / 2)
            y1 = y0 + nh
        return (max(0, x0), max(0, y0), min(self.cfg.width - 1, x1), min(self.cfg.height - 1, y1))
