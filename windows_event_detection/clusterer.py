from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import math
import time
import numpy as np

from config import DetectionConfig
from event_types import DynamicObject, Track


def _cos_sim(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    a = math.hypot(v1[0], v1[1])
    b = math.hypot(v2[0], v2[1])
    if a < 1e-6 or b < 1e-6:
        return 1.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (a * b)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


class DynamicObjectClusterer:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.next_obj_id = 1
        self.objects: Dict[int, DynamicObject] = {}

    def _track_ok(self, t: Track) -> bool:
        return t.confidence >= self.cfg.min_track_confidence_for_cluster or len(t.points) >= self.cfg.min_track_len

    def _pair_compatible(self, a: Track, b: Track) -> bool:
        pa, pb = a.points[-1], b.points[-1]
        dist = math.hypot(pa.x - pb.x, pa.y - pb.y)
        if dist > self.cfg.cluster_spatial_threshold:
            return False
        if abs(pa.t - pb.t) > self.cfg.cluster_time_threshold_us:
            return False
        if not (self._track_ok(a) and self._track_ok(b)):
            return False

        sa = math.hypot(a.velocity[0], a.velocity[1])
        sb = math.hypot(b.velocity[0], b.velocity[1])
        if abs(sa - sb) > self.cfg.cluster_speed_diff_threshold:
            return False

        both_low_speed = sa < self.cfg.min_motion_speed and sb < self.cfg.min_motion_speed
        if both_low_speed:
            return dist <= self.cfg.cluster_spatial_threshold * self.cfg.low_speed_spatial_scale and min(a.residual, b.residual) >= self.cfg.min_object_residual_mean
        if sa >= self.cfg.min_motion_speed and sb >= self.cfg.min_motion_speed:
            return _cos_sim(a.velocity, b.velocity) >= self.cfg.cluster_velocity_cos_threshold
        return dist <= self.cfg.cluster_spatial_threshold * self.cfg.low_speed_spatial_scale

    def _build_clusters(self, tracks: List[Track]) -> List[List[Track]]:
        grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        cs = max(10, int(self.cfg.cluster_grid_cell_size))
        for i, t in enumerate(tracks):
            if not t.points:
                continue
            p = t.points[-1]
            grid[(int(p.x) // cs, int(p.y) // cs)].append(i)

        parent = list(range(len(tracks)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, t in enumerate(tracks):
            if not t.points:
                continue
            p = t.points[-1]
            gx, gy = int(p.x) // cs, int(p.y) // cs
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((gx + dx, gy + dy), []):
                        if j <= i:
                            continue
                        if self._pair_compatible(t, tracks[j]):
                            union(i, j)

        comp: Dict[int, List[Track]] = defaultdict(list)
        for i, t in enumerate(tracks):
            if t.points:
                comp[find(i)].append(t)
        return [c for c in comp.values() if len(c) >= self.cfg.cluster_min_tracks]

    def _cluster_to_object(self, cluster: List[Track], events: np.ndarray) -> DynamicObject:
        xs = [float(t.points[-1].x) for t in cluster]
        ys = [float(t.points[-1].y) for t in cluster]
        x0, x1 = int(min(xs) - 8), int(max(xs) + 8)
        y0, y1 = int(min(ys) - 8), int(max(ys) + 8)
        bbox = (max(0, x0), max(0, y0), x1, y1)
        vv = np.array([t.velocity for t in cluster], dtype=np.float32)
        speeds = np.linalg.norm(vv, axis=1)
        dirs = vv / np.maximum(speeds[:, None], 1e-6)
        dir_mean = np.mean(dirs, axis=0)
        dir_mean_norm = np.linalg.norm(dir_mean)
        direction_std = float(1.0 - np.clip(dir_mean_norm, 0.0, 1.0))
        speed_mean = float(np.mean(speeds))
        speed_std = float(np.std(speeds))
        residual_mean = float(np.mean([t.residual for t in cluster]))
        avg_len = float(np.mean([len(t.points) for t in cluster]))
        ex = events['x'].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
        ey = events['y'].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
        support = int(np.count_nonzero((ex >= bbox[0]) & (ex <= bbox[2]) & (ey >= bbox[1]) & (ey <= bbox[3]))) if len(events) else 0
        aspect = max((bbox[2]-bbox[0]) / max(1.0, (bbox[3]-bbox[1])), (bbox[3]-bbox[1]) / max(1.0, (bbox[2]-bbox[0])))
        consistency = 0.35 * max(0.0, 1.0 - direction_std / max(self.cfg.max_direction_std_for_object, 1e-3))
        consistency += 0.25 * max(0.0, 1.0 - speed_std / max(self.cfg.max_speed_std_for_object, 1e-3))
        consistency += 0.15 * min(1.0, len(cluster) / max(self.cfg.min_object_tracks, 1))
        consistency += 0.15 * min(1.0, support / max(self.cfg.min_object_event_support, 1))
        consistency += 0.10 * min(1.0, residual_mean / max(self.cfg.min_object_residual_mean, 1e-3))
        if aspect > self.cfg.max_bbox_aspect_ratio:
            consistency *= 0.6
        avg_v = np.mean(vv, axis=0)
        return DynamicObject(
            object_id=0,
            last_bbox=bbox,
            last_center=(float(np.mean(xs)), float(np.mean(ys))),
            velocity=(float(avg_v[0]), float(avg_v[1])),
            missed_count=0,
            confidence=float(np.clip(np.mean([t.confidence for t in cluster]) * consistency, 0.05, 1.0)),
            predicted=False,
            track_ids=[t.track_id for t in cluster],
            speed_mean=speed_mean,
            speed_std=speed_std,
            direction_std=direction_std,
            motion_consistency_score=float(np.clip(consistency, 0.0, 1.0)),
            track_count=len(cluster),
            event_support_count=support,
            residual_mean=residual_mean,
            avg_track_length=avg_len,
        )

    def _can_merge_objects(self, a: DynamicObject, b: DynamicObject) -> bool:
        if _cos_sim(a.velocity, b.velocity) < self.cfg.object_merge_velocity_cos_threshold:
            return False
        if abs(a.speed_mean - b.speed_mean) > self.cfg.object_merge_speed_diff_threshold:
            return False
        close = _bbox_iou(a.last_bbox, b.last_bbox) > self.cfg.object_merge_iou_threshold
        if not close:
            cd = math.hypot(a.last_center[0] - b.last_center[0], a.last_center[1] - b.last_center[1])
            close = cd <= self.cfg.object_merge_center_distance
        if not close:
            return False
        return min(a.motion_consistency_score, b.motion_consistency_score) >= self.cfg.min_motion_consistency_score

    def _filter_object(self, obj: DynamicObject) -> bool:
        w = obj.last_bbox[2] - obj.last_bbox[0]
        h = obj.last_bbox[3] - obj.last_bbox[1]
        area = w * h
        aspect = max(w / max(1.0, h), h / max(1.0, w))
        if area < self.cfg.min_bbox_area or w < self.cfg.min_bbox_width or h < self.cfg.min_bbox_height:
            return False
        if obj.track_count < self.cfg.min_object_tracks or obj.event_support_count < self.cfg.min_object_event_support:
            return False
        if obj.motion_consistency_score < self.cfg.min_motion_consistency_score:
            return False
        if obj.speed_std > self.cfg.max_speed_std_for_object or obj.direction_std > self.cfg.max_direction_std_for_object:
            return False
        if obj.residual_mean < self.cfg.min_object_residual_mean or obj.avg_track_length < self.cfg.min_object_avg_track_length:
            return False
        if aspect > self.cfg.max_bbox_aspect_ratio and obj.track_count <= self.cfg.min_object_tracks + 1:
            return False
        return True

    def update(self, dynamic_tracks: List[Track], uncertain_tracks: List[Track], events: np.ndarray) -> Tuple[List[DynamicObject], int, Dict[str, float]]:
        t0 = time.perf_counter()
        candidates = list(dynamic_tracks)
        if len(candidates) < self.cfg.cluster_min_tracks:
            candidates += [t for t in uncertain_tracks if t.residual >= 1.2 * self.cfg.static_residual_threshold]
        clusters = self._build_clusters(candidates)
        t1 = time.perf_counter()

        raw_objects = [self._cluster_to_object(c, events) for c in clusters]

        merged: List[DynamicObject] = []
        for obj in raw_objects:
            done = False
            for i, mo in enumerate(merged):
                if self._can_merge_objects(mo, obj):
                    track_ids = list(set(mo.track_ids + obj.track_ids))
                    # lightweight merge recompute
                    bx = (min(mo.last_bbox[0], obj.last_bbox[0]), min(mo.last_bbox[1], obj.last_bbox[1]), max(mo.last_bbox[2], obj.last_bbox[2]), max(mo.last_bbox[3], obj.last_bbox[3]))
                    merged[i] = DynamicObject(
                        0, bx,
                        ((mo.last_center[0] + obj.last_center[0]) / 2.0, (mo.last_center[1] + obj.last_center[1]) / 2.0),
                        ((mo.velocity[0] + obj.velocity[0]) / 2.0, (mo.velocity[1] + obj.velocity[1]) / 2.0),
                        0,
                        max(mo.confidence, obj.confidence),
                        False,
                        track_ids,
                        speed_mean=(mo.speed_mean + obj.speed_mean) / 2.0,
                        speed_std=max(mo.speed_std, obj.speed_std),
                        direction_std=max(mo.direction_std, obj.direction_std),
                        motion_consistency_score=min(mo.motion_consistency_score, obj.motion_consistency_score),
                        track_count=mo.track_count + obj.track_count,
                        event_support_count=mo.event_support_count + obj.event_support_count,
                        residual_mean=(mo.residual_mean + obj.residual_mean) / 2.0,
                        avg_track_length=(mo.avg_track_length + obj.avg_track_length) / 2.0,
                    )
                    done = True
                    break
            if not done:
                merged.append(obj)
        t2 = time.perf_counter()

        filtered = [o for o in merged if self._filter_object(o)]

        detected_objs: List[DynamicObject] = []
        used_prev = set()
        for obj in filtered:
            best_id, best_score = -1, -1.0
            for oid, prev in self.objects.items():
                iou = _bbox_iou(obj.last_bbox, prev.last_bbox)
                cdist = math.hypot(obj.last_center[0] - prev.last_center[0], obj.last_center[1] - prev.last_center[1])
                if iou < self.cfg.object_match_iou_threshold and cdist > self.cfg.object_match_center_distance:
                    continue
                vcos = _cos_sim(obj.velocity, prev.velocity)
                if vcos < self.cfg.object_match_velocity_cos_threshold:
                    continue
                if abs(obj.speed_mean - prev.speed_mean) > self.cfg.object_match_speed_diff_threshold:
                    continue
                score = iou * 2.0 + vcos - cdist / max(self.cfg.object_match_center_distance, 1.0)
                if score > best_score:
                    best_score, best_id = score, oid
            if best_id >= 0:
                prev = self.objects[best_id]
                a = self.cfg.bbox_smooth_alpha
                x0 = int(a * prev.last_bbox[0] + (1 - a) * obj.last_bbox[0])
                y0 = int(a * prev.last_bbox[1] + (1 - a) * obj.last_bbox[1])
                x1 = int(a * prev.last_bbox[2] + (1 - a) * obj.last_bbox[2])
                y1 = int(a * prev.last_bbox[3] + (1 - a) * obj.last_bbox[3])
                va = self.cfg.velocity_smooth_alpha_object
                obj.object_id = best_id
                obj.last_bbox = (x0, y0, x1, y1)
                obj.velocity = (va * prev.velocity[0] + (1 - va) * obj.velocity[0], va * prev.velocity[1] + (1 - va) * obj.velocity[1])
                obj.missed_count = 0
                used_prev.add(best_id)
            else:
                obj.object_id = self.next_obj_id
                self.next_obj_id += 1
            self.objects[obj.object_id] = obj
            detected_objs.append(obj)

        predicted_count = 0
        ex = events['x'].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
        ey = events['y'].astype(np.int32) if len(events) else np.array([], dtype=np.int32)
        for oid, prev in list(self.objects.items()):
            if oid in {o.object_id for o in detected_objs}:
                continue
            prev.missed_count += 1
            if prev.missed_count > self.cfg.predicted_max_age:
                del self.objects[oid]
                continue
            px = prev.last_center[0] + prev.velocity[0]
            py = prev.last_center[1] + prev.velocity[1]
            support = 0
            if len(events):
                rr = self.cfg.predicted_event_support_radius
                support = int(np.count_nonzero((ex >= px - rr) & (ex <= px + rr) & (ey >= py - rr) & (ey <= py + rr)))
            if support < self.cfg.predicted_min_event_support:
                del self.objects[oid]
                continue
            w = prev.last_bbox[2] - prev.last_bbox[0]
            h = prev.last_bbox[3] - prev.last_bbox[1]
            prev.last_center = (px, py)
            prev.last_bbox = (int(px - w / 2), int(py - h / 2), int(px + w / 2), int(py + h / 2))
            prev.predicted = True
            prev.event_support_count = support
            detected_objs.append(prev)
            predicted_count += 1

        t3 = time.perf_counter()
        debug = {
            'raw_cluster_count': len(clusters),
            'merged_object_count': len(merged),
            'filtered_object_count': len(filtered),
            'final_object_count': len(detected_objs),
            'cluster_time_ms': (t1 - t0) * 1000.0,
            'object_merge_time_ms': (t2 - t1) * 1000.0,
            'object_tracking_time_ms': (t3 - t2) * 1000.0,
        }
        return detected_objs, predicted_count, debug
