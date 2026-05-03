from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np

from config import DetectionConfig
from event_types import DynamicObject, Track


BBox = Tuple[int, int, int, int]


def _cos_sim(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    a = math.hypot(v1[0], v1[1])
    b = math.hypot(v2[0], v2[1])
    if a < 1e-6 or b < 1e-6:
        return 1.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (a * b)


def _speed(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _bbox_iou(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    iw, ih = max(0, x1 - x0), max(0, y1 - y0)
    inter = iw * ih
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def _bbox_center(a: BBox) -> Tuple[float, float]:
    return ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5)


def _center_dist(a: BBox, b: BBox) -> float:
    ca = _bbox_center(a)
    cb = _bbox_center(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def _edge_dist(a: BBox, b: BBox) -> float:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _bbox_union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class DynamicObjectClusterer:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.next_obj_id = 1
        self.objects: Dict[int, DynamicObject] = {}
        self.last_stats: Dict[str, int] = {}

    def _build_clusters(self, tracks: List[Track]) -> List[List[Track]]:
        clusters: List[List[Track]] = []
        for t in tracks:
            if not t.points:
                continue
            assigned = False
            for c in clusters:
                pt, cpt = t.points[-1], c[0].points[-1]
                d = math.hypot(pt.x - cpt.x, pt.y - cpt.y)
                if d <= self.cfg.cluster_spatial_threshold and _cos_sim(t.velocity, c[0].velocity) >= self.cfg.cluster_velocity_cos_threshold:
                    c.append(t)
                    assigned = True
                    break
            if not assigned:
                clusters.append([t])
        return [c for c in clusters if len(c) >= self.cfg.cluster_min_tracks]

    def _cluster_to_object(self, cluster: List[Track]) -> DynamicObject:
        xs = np.array([t.points[-1].x for t in cluster], dtype=np.float32)
        ys = np.array([t.points[-1].y for t in cluster], dtype=np.float32)
        vs = np.array([t.velocity for t in cluster], dtype=np.float32)
        speeds = np.linalg.norm(vs, axis=1) if len(vs) else np.array([0.0], dtype=np.float32)
        dirs = np.arctan2(vs[:, 1], vs[:, 0]) if len(vs) else np.array([0.0], dtype=np.float32)
        bbox = (int(xs.min() - 8), int(ys.min() - 8), int(xs.max() + 8), int(ys.max() + 8))
        vv = np.mean(vs, axis=0)
        direction_std = float(np.std(np.unwrap(dirs))) if len(dirs) > 1 else 0.0
        speed_std = float(np.std(speeds))
        motion_consistency = float(np.clip(1.0 - 0.2 * direction_std - 0.1 * speed_std, 0.0, 1.0))
        return DynamicObject(
            object_id=0,
            last_bbox=bbox,
            last_center=(float(xs.mean()), float(ys.mean())),
            velocity=(float(vv[0]), float(vv[1])),
            missed_count=0,
            confidence=float(np.clip(np.mean([t.confidence for t in cluster]) * motion_consistency, 0.2, 1.0)),
            predicted=False,
            track_ids=[t.track_id for t in cluster],
            speed_mean=float(np.mean(speeds)),
            speed_std=speed_std,
            direction_std=direction_std,
            motion_consistency_score=motion_consistency,
            track_count=len(cluster),
        )

    def _merge_objects(self, objs: List[DynamicObject], iou_th: float, edge_th: float, center_th: float, vcos_th: float, sdiff_th: float) -> Tuple[List[DynamicObject], int]:
        merged = 0
        changed = True
        while changed and len(objs) > 1:
            changed = False
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i], objs[j]
                    spatial = (_bbox_iou(a.last_bbox, b.last_bbox) > iou_th or _edge_dist(a.last_bbox, b.last_bbox) < edge_th or _center_dist(a.last_bbox, b.last_bbox) < center_th)
                    motion = (_cos_sim(a.velocity, b.velocity) > vcos_th and abs(_speed(a.velocity) - _speed(b.velocity)) < sdiff_th)
                    if spatial and motion:
                        merged += 1
                        nn = DynamicObject(
                            object_id=a.object_id,
                            last_bbox=_bbox_union(a.last_bbox, b.last_bbox),
                            last_center=((a.last_center[0] + b.last_center[0]) * 0.5, (a.last_center[1] + b.last_center[1]) * 0.5),
                            velocity=((a.velocity[0] + b.velocity[0]) * 0.5, (a.velocity[1] + b.velocity[1]) * 0.5),
                            missed_count=min(a.missed_count, b.missed_count),
                            confidence=max(a.confidence, b.confidence),
                            predicted=a.predicted and b.predicted,
                            track_ids=sorted(set(a.track_ids + b.track_ids)),
                            speed_mean=(a.speed_mean + b.speed_mean) * 0.5,
                            speed_std=(a.speed_std + b.speed_std) * 0.5,
                            direction_std=(a.direction_std + b.direction_std) * 0.5,
                            motion_consistency_score=max(a.motion_consistency_score, b.motion_consistency_score),
                            track_count=a.track_count + b.track_count,
                        )
                        objs = objs[:i] + [nn] + objs[i + 1:j] + objs[j + 1:]
                        changed = True
                        break
                if changed:
                    break
        return objs, merged

    def _predict_bbox(self, obj: DynamicObject) -> BBox:
        x0, y0, x1, y1 = obj.last_bbox
        dx, dy = obj.velocity
        return (int(x0 + dx * 0.01), int(y0 + dy * 0.01), int(x1 + dx * 0.01), int(y1 + dy * 0.01))

    def _fuse_bbox(self, det: BBox, pred: BBox, prev: BBox) -> BBox:
        base = _bbox_union(det, pred) if self.cfg.bbox_use_union_with_prediction else det
        w, h = max(1, base[2] - base[0]), max(1, base[3] - base[1])
        px = max(self.cfg.bbox_min_padding, int(w * self.cfg.bbox_padding_ratio_x))
        py = max(self.cfg.bbox_min_padding, int(h * self.cfg.bbox_padding_ratio_y))
        pad = (base[0] - px, base[1] - py, base[2] + px, base[3] + py)
        a = self.cfg.bbox_smooth_alpha
        return tuple(int((1 - a) * p + a * n) for p, n in zip(prev, pad))

    def update(self, dynamic_tracks: List[Track], uncertain_tracks: List[Track], events: np.ndarray) -> Tuple[List[DynamicObject], int]:
        candidates = list(dynamic_tracks)
        if len(candidates) < self.cfg.cluster_min_tracks:
            candidates += [t for t in uncertain_tracks if t.residual >= 1.2 * self.cfg.static_residual_threshold]

        detected = [self._cluster_to_object(c) for c in self._build_clusters(candidates)]
        detected, same_merge = self._merge_objects(detected, self.cfg.same_frame_merge_iou_threshold, self.cfg.same_frame_merge_edge_distance, self.cfg.same_frame_merge_center_distance, self.cfg.same_frame_merge_velocity_cos_threshold, self.cfg.same_frame_merge_speed_diff_threshold)

        finals: List[DynamicObject] = []
        matched_prev = set()
        matched_det = set()
        matched_with_prediction = 0

        prev_items = list(self.objects.items())
        for di, d in enumerate(detected):
            best = None
            best_score = -1.0
            for oid, prev in prev_items:
                if oid in matched_prev:
                    continue
                pred_bbox = self._predict_bbox(prev)
                spatial = (_bbox_iou(d.last_bbox, pred_bbox) > self.cfg.object_match_iou_threshold or _center_dist(d.last_bbox, pred_bbox) < self.cfg.object_match_center_distance or _edge_dist(d.last_bbox, pred_bbox) < self.cfg.object_match_edge_distance)
                motion = (_cos_sim(d.velocity, prev.velocity) > self.cfg.object_match_velocity_cos_threshold and abs(_speed(d.velocity) - _speed(prev.velocity)) < self.cfg.object_match_speed_diff_threshold)
                if spatial and motion:
                    sc = _bbox_iou(d.last_bbox, pred_bbox) + _cos_sim(d.velocity, prev.velocity)
                    if sc > best_score:
                        best_score, best = sc, (oid, prev, pred_bbox)
            if best is None:
                continue
            oid, prev, pred_bbox = best
            matched_prev.add(oid)
            matched_det.add(di)
            matched_with_prediction += 1
            fused_bbox = self._fuse_bbox(d.last_bbox, pred_bbox, prev.last_bbox)
            vv = (prev.velocity[0] * 0.6 + d.velocity[0] * 0.4, prev.velocity[1] * 0.6 + d.velocity[1] * 0.4)
            updated = DynamicObject(object_id=oid, last_bbox=fused_bbox, last_center=_bbox_center(fused_bbox), velocity=vv, missed_count=0, confidence=max(prev.confidence, d.confidence), predicted=False, track_ids=sorted(set(prev.track_ids + d.track_ids)), speed_mean=(prev.speed_mean + d.speed_mean) * 0.5, speed_std=(prev.speed_std + d.speed_std) * 0.5, direction_std=(prev.direction_std + d.direction_std) * 0.5, motion_consistency_score=max(prev.motion_consistency_score, d.motion_consistency_score), track_count=max(prev.track_count, d.track_count))
            self.objects[oid] = updated
            finals.append(updated)

        for i, d in enumerate(detected):
            if i in matched_det:
                continue
            d.object_id = self.next_obj_id
            self.next_obj_id += 1
            self.objects[d.object_id] = d
            finals.append(d)

        predicted_count = 0
        suppressed_predicted = 0
        for oid, prev in list(self.objects.items()):
            if oid in matched_prev or any(f.object_id == oid for f in finals):
                continue
            prev.missed_count += 1
            if prev.missed_count > self.cfg.predicted_max_age:
                del self.objects[oid]
                continue
            pb = self._predict_bbox(prev)
            ex = events["x"].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
            ey = events["y"].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
            support = int(np.count_nonzero((ex >= pb[0]) & (ex <= pb[2]) & (ey >= pb[1]) & (ey <= pb[3]))) if len(events) else 0
            close_to_final = any(_bbox_iou(pb, f.last_bbox) > 0.01 or _center_dist(pb, f.last_bbox) < self.cfg.object_match_center_distance or _edge_dist(pb, f.last_bbox) < self.cfg.object_match_edge_distance for f in finals)
            if support >= self.cfg.object_min_event_support and not close_to_final:
                prev.last_bbox = pb
                prev.last_center = _bbox_center(pb)
                prev.predicted = True
                self.objects[oid] = prev
                finals.append(prev)
                predicted_count += 1
            else:
                suppressed_predicted += 1

        finals, dedup = self._merge_objects(finals, self.cfg.final_nms_iou_threshold, self.cfg.final_nms_edge_distance, self.cfg.final_nms_center_distance, 0.3, 999.0)
        self.last_stats = {
            "same_frame_merged_count": same_merge,
            "matched_with_prediction_count": matched_with_prediction,
            "suppressed_predicted_count": suppressed_predicted,
            "final_duplicate_removed_count": dedup,
        }
        return finals, predicted_count
