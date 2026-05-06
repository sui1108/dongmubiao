from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np
import cv2

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
        self.id_switch_count_estimate = 0
        self.bbox_shrink_suppressed_count = 0
        self.event_completion_used_count = 0
        self.total_predicted_center_error = 0.0
        self.predicted_center_error_count = 0

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

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter / (aa + ab - inter))

    def _clip_bbox(self, bbox: Tuple[int, int, int, int], events: np.ndarray) -> Tuple[int, int, int, int]:
        if len(events):
            w = int(np.max(events["x"])) + 1
            h = int(np.max(events["y"])) + 1
        else:
            w, h = 640, 480
        return (max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3]))

    def _event_support(self, bbox: Tuple[int, int, int, int], events: np.ndarray) -> int:
        if len(events) == 0:
            return 0
        ex = events["x"].astype(np.int32)
        ey = events["y"].astype(np.int32)
        return int(np.count_nonzero((ex >= bbox[0]) & (ex <= bbox[2]) & (ey >= bbox[1]) & (ey <= bbox[3])))

    def _measurement_bbox(self, primary_bbox: Tuple[int, int, int, int], hist_bbox: Tuple[int, int, int, int], child_boxes: List[Tuple[int, int, int, int]], events: np.ndarray) -> Tuple[Tuple[int, int, int, int], bool]:
        roi = (min(primary_bbox[0], hist_bbox[0]), min(primary_bbox[1], hist_bbox[1]), max(primary_bbox[2], hist_bbox[2]), max(primary_bbox[3], hist_bbox[3]))
        support_bbox = primary_bbox
        used_completion = False
        if len(events):
            ex = events["x"].astype(np.int32)
            ey = events["y"].astype(np.int32)
            in_roi = (ex >= roi[0]) & (ex <= roi[2]) & (ey >= roi[1]) & (ey <= roi[3])
            if np.count_nonzero(in_roi) >= self.cfg.object_min_event_support:
                mask = np.zeros((roi[3] - roi[1] + 1, roi[2] - roi[0] + 1), dtype=np.uint8)
                xs = ex[in_roi] - roi[0]
                ys = ey[in_roi] - roi[1]
                mask[ys, xs] = 255
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                nlabels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if nlabels > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    x = int(stats[largest, cv2.CC_STAT_LEFT] + roi[0])
                    y = int(stats[largest, cv2.CC_STAT_TOP] + roi[1])
                    w = int(stats[largest, cv2.CC_STAT_WIDTH])
                    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
                    support_bbox = (x, y, x + w, y + h)
                    used_completion = True
        if child_boxes:
            cx0 = min(b[0] for b in child_boxes)
            cy0 = min(b[1] for b in child_boxes)
            cx1 = max(b[2] for b in child_boxes)
            cy1 = max(b[3] for b in child_boxes)
            support_bbox = (min(support_bbox[0], cx0), min(support_bbox[1], cy0), max(support_bbox[2], cx1), max(support_bbox[3], cy1))
        return self._clip_bbox(support_bbox, events), used_completion

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
            detected_objs.append(DynamicObject(0, bbox, (cx, cy), (float(vv[0]), float(vv[1])), 1, 0, conf, 0.0, False, [t.track_id for t in cluster]))

        predicted_count = 0
        used = set()
        for obj in detected_objs:
            best_id, best_score = -1, -1.0
            for oid, prev in self.objects.items():
                pc = (prev.last_center[0] + prev.velocity[0] * 0.01, prev.last_center[1] + prev.velocity[1] * 0.01)
                pred_d = math.hypot(obj.last_center[0] - pc[0], obj.last_center[1] - pc[1])
                center_d = math.hypot(obj.last_center[0] - prev.last_center[0], obj.last_center[1] - prev.last_center[1])
                edge_d = math.hypot(obj.last_bbox[0] - prev.last_bbox[0], obj.last_bbox[1] - prev.last_bbox[1]) + math.hypot(obj.last_bbox[2] - prev.last_bbox[2], obj.last_bbox[3] - prev.last_bbox[3])
                iou = self._iou(obj.last_bbox, prev.last_bbox)
                vel_sim = 0.5 * (_cos_sim(obj.velocity, prev.velocity) + 1.0)
                cur_support = self._event_support(obj.last_bbox, events)
                prev_support = max(1.0, prev.recent_event_support)
                support_overlap = min(cur_support, prev_support) / max(cur_support, prev_support)
                stable_bonus = min(0.2, prev.age / 100.0)
                score = (
                    0.30 * iou
                    + 0.20 * max(0.0, 1.0 - center_d / (self.cfg.cluster_spatial_threshold * 2.0))
                    + 0.15 * max(0.0, 1.0 - edge_d / (self.cfg.cluster_spatial_threshold * 6.0))
                    + 0.10 * vel_sim
                    + 0.15 * max(0.0, 1.0 - pred_d / (self.cfg.cluster_spatial_threshold * 2.0))
                    + 0.10 * support_overlap
                    + stable_bonus
                )
                if score > best_score:
                    best_score, best_id = score, oid
            if best_id >= 0 and best_score > 0.35:
                obj.object_id = best_id
                prev = self.objects[best_id]
                measurement_bbox, used_completion = self._measurement_bbox(obj.last_bbox, prev.last_bbox, [obj.last_bbox], events)
                if used_completion:
                    self.event_completion_used_count += 1
                old_area = max(1, (prev.last_bbox[2] - prev.last_bbox[0]) * (prev.last_bbox[3] - prev.last_bbox[1]))
                new_area = max(1, (measurement_bbox[2] - measurement_bbox[0]) * (measurement_bbox[3] - measurement_bbox[1]))
                if new_area < old_area * 0.5:
                    support_hist = self._event_support(prev.last_bbox, events)
                    if support_hist >= self.cfg.object_min_event_support:
                        measurement_bbox = prev.last_bbox
                        self.bbox_shrink_suppressed_count += 1
                    else:
                        ratio = max(0.8, new_area / old_area)
                        cx, cy = obj.last_center
                        pw = max(2, int((prev.last_bbox[2] - prev.last_bbox[0]) * ratio))
                        ph = max(2, int((prev.last_bbox[3] - prev.last_bbox[1]) * ratio))
                        measurement_bbox = (int(cx - pw / 2), int(cy - ph / 2), int(cx + pw / 2), int(cy + ph / 2))
                        self.bbox_shrink_suppressed_count += 1
                obj.last_bbox = self._clip_bbox(measurement_bbox, events)
                obj.last_center = ((obj.last_bbox[0] + obj.last_bbox[2]) / 2.0, (obj.last_bbox[1] + obj.last_bbox[3]) / 2.0)
                obj.age = prev.age + 1
                obj.missed_count = 0
                obj.confidence = min(1.0, 0.8 * prev.confidence + 0.2 * obj.confidence + 0.03)
                obj.recent_event_support = 0.8 * prev.recent_event_support + 0.2 * self._event_support(obj.last_bbox, events)
                used.add(best_id)
            else:
                obj.object_id = self.next_obj_id
                self.next_obj_id += 1
                obj.recent_event_support = float(self._event_support(obj.last_bbox, events))
                if best_id >= 0:
                    self.id_switch_count_estimate += 1
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
                    prev.age += 1
                    prev.recent_event_support = 0.9 * prev.recent_event_support + 0.1 * support
                    detected_objs.append(prev)
                    self.total_predicted_center_error += math.hypot(px - prev.last_center[0], py - prev.last_center[1])
                    self.predicted_center_error_count += 1
                    predicted_count += 1
                else:
                    del self.objects[oid]
            else:
                del self.objects[oid]

        return detected_objs, predicted_count
