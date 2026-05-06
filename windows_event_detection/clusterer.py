from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import math
import numpy as np

from config import DetectionConfig
from event_types import DynamicObject, Track


class DynamicObjectClusterer:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.next_obj_id = 1
        self.tracked_objects: Dict[int, DynamicObject] = {}
        self._last_raw_clusters: List[DynamicObject] = []
        self._last_primary_objects: List[DynamicObject] = []
        self.last_primary_merge_ms = 0.0
        self.direction_reverse_reuse_count = 0
        self.rejected_new_object_count = 0
        self.roi_match_count = 0
        self.roi_outside_reject_count = 0
        self.tight_bbox_shrink_count = 0

    def _bbox_iou(self, a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(ua, 1e-6)

    def _edge_distance(self, a, b):
        return max(0, max(a[0]-b[2], b[0]-a[2], a[1]-b[3], b[1]-a[3]))

    def _event_support(self, events, bbox):
        if len(events) == 0:
            return 0
        ex = events['x'].astype(np.int32); ey = events['y'].astype(np.int32)
        return int(np.count_nonzero((ex >= bbox[0]) & (ex <= bbox[2]) & (ey >= bbox[1]) & (ey <= bbox[3])))

    def _tighten_bbox(self, bbox, events):
        support = self._event_support(events, bbox)
        area = max(1, (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        density = support / area
        if support == 0:
            return bbox, density, 1.0
        ex = events['x'].astype(np.int32); ey = events['y'].astype(np.int32)
        m = (ex >= bbox[0]) & (ex <= bbox[2]) & (ey >= bbox[1]) & (ey <= bbox[3])
        tx0, tx1 = int(np.min(ex[m])), int(np.max(ex[m]))
        ty0, ty1 = int(np.min(ey[m])), int(np.max(ey[m]))
        tight_area = max(1, (tx1-tx0)*(ty1-ty0))
        ratio = tight_area / area
        if ratio < self.cfg.tight_bbox_shrink_threshold:
            self.tight_bbox_shrink_count += 1
            p = self.cfg.tight_bbox_padding
            return (max(0, tx0-p), max(0, ty0-p), min(self.cfg.width-1, tx1+p), min(self.cfg.height-1, ty1+p)), density, ratio
        return bbox, density, ratio

    def _expand_roi(self, bbox):
        cx = (bbox[0]+bbox[2])/2; cy = (bbox[1]+bbox[3])/2
        w = (bbox[2]-bbox[0])*self.cfg.roi_expand_ratio; h = (bbox[3]-bbox[1])*self.cfg.roi_expand_ratio
        return (max(0, int(cx-w/2)), max(0, int(cy-h/2)), min(self.cfg.width-1, int(cx+w/2)), min(self.cfg.height-1, int(cy+h/2)))

    def _within(self, p, b):
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

    def _build_raw(self, tracks: List[Track], events: np.ndarray) -> List[DynamicObject]:
        cell = self.cfg.cluster_grid_cell_size
        grid: Dict[Tuple[int, int], List[Track]] = defaultdict(list)
        for t in tracks:
            if t.points:
                p = t.points[-1]; grid[(int(p.x // cell), int(p.y // cell))].append(t)
        used, clusters = set(), []
        for t in tracks:
            if t.track_id in used or not t.points: continue
            seed = [t]; used.add(t.track_id)
            sx, sy = int(t.points[-1].x // cell), int(t.points[-1].y // cell)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for n in grid.get((sx+dx, sy+dy), []):
                        if n.track_id in used or not n.points: continue
                        if math.hypot(t.points[-1].x-n.points[-1].x, t.points[-1].y-n.points[-1].y) <= self.cfg.cluster_spatial_threshold:
                            seed.append(n); used.add(n.track_id)
            if len(seed) >= self.cfg.cluster_min_tracks:
                xs=[int(k.points[-1].x) for k in seed]; ys=[int(k.points[-1].y) for k in seed]
                bb=(max(0,min(xs)-8), max(0,min(ys)-8), max(xs)+8, max(ys)+8)
                bb,density,ratio = self._tighten_bbox(bb, events)
                vel=np.mean(np.array([k.velocity for k in seed],dtype=np.float32), axis=0)
                es = self._event_support(events, bb)
                clusters.append(DynamicObject(0,bb,((bb[0]+bb[2])/2,(bb[1]+bb[3])/2),(float(vel[0]),float(vel[1])),0,float(np.mean([k.confidence for k in seed])),False,[k.track_id for k in seed],kind='raw_cluster',event_support=es,density=density,bbox_density=density,tight_bbox_area_ratio=ratio))
        return clusters

    def update(self, dynamic_tracks: List[Track], uncertain_tracks: List[Track], events: np.ndarray):
        raw = self._build_raw(list(dynamic_tracks), events)
        self._last_raw_clusters = raw
        self.rejected_new_object_count = self.roi_match_count = self.roi_outside_reject_count = 0

        unmatched_prev = set(self.tracked_objects.keys())
        matched_ids = set()
        detections = []
        for r in raw:
            best_id, best_score, reverse_motion = None, -1e9, False
            in_roi_ids = [oid for oid, prev in self.tracked_objects.items() if prev.roi_bbox and self._within(r.last_center, prev.roi_bbox)]
            if in_roi_ids:
                self.roi_match_count += 1
            candidate_ids = in_roi_ids if in_roi_ids else list(self.tracked_objects.keys())
            for oid in candidate_ids:
                prev = self.tracked_objects[oid]
                iou = self._bbox_iou(r.last_bbox, prev.last_bbox)
                cd = math.hypot(r.last_center[0]-prev.last_center[0], r.last_center[1]-prev.last_center[1])
                ed = self._edge_distance(r.last_bbox, prev.last_bbox)
                a0 = max(1,(r.last_bbox[2]-r.last_bbox[0])*(r.last_bbox[3]-r.last_bbox[1]))
                a1 = max(1,(prev.last_bbox[2]-prev.last_bbox[0])*(prev.last_bbox[3]-prev.last_bbox[1]))
                size_sim = min(a0,a1)/max(a0,a1)
                es_sim = min(r.event_support, prev.event_support)/max(1,max(r.event_support, prev.event_support))
                vm0 = math.hypot(*r.velocity); vm1 = math.hypot(*prev.velocity)
                mag = min(vm0,vm1)/max(1e-6,max(vm0,vm1))
                cos = 0.0 if vm0*vm1 < 1e-6 else (r.velocity[0]*prev.velocity[0]+r.velocity[1]*prev.velocity[1])/(vm0*vm1)
                score = 2.0*iou - cd/120.0 - ed/100.0 + size_sim + 0.7*es_sim + 0.5*mag + 0.2*cos
                if cos < 0 and cd < self.cfg.primary_merge_center_distance*0.4 and (iou > 0.05 or ed < 12):
                    score += self.cfg.reverse_motion_tolerance
                if score > best_score:
                    best_score, best_id, reverse_motion = score, oid, cos < 0
            matched = best_id is not None and best_score > -0.8
            if matched:
                prev = self.tracked_objects[best_id]
                r.object_id = best_id; r.hits = prev.hits + 1; r.age = prev.age + 1; r.missed_count = 0
                r.state = 'confirmed' if (prev.state == 'confirmed' or r.hits >= self.cfg.tentative_confirm_frames) else 'tentative'
                r.reused_id = True; r.reverse_motion = reverse_motion
                if reverse_motion and r.state == 'confirmed': self.direction_reverse_reuse_count += 1
                r.lost_recovered_count = prev.lost_recovered_count + (1 if prev.state == 'lost' else 0)
                matched_ids.add(best_id); unmatched_prev.discard(best_id)
            else:
                area = max(1,(r.last_bbox[2]-r.last_bbox[0])*(r.last_bbox[3]-r.last_bbox[1]))
                ar = max((r.last_bbox[2]-r.last_bbox[0])/max(1,(r.last_bbox[3]-r.last_bbox[1])), (r.last_bbox[3]-r.last_bbox[1])/max(1,(r.last_bbox[2]-r.last_bbox[0])))
                outside_roi = len(in_roi_ids) == 0 and len(self.tracked_objects) > 0
                min_tracks = int(math.ceil(self.cfg.min_new_object_tracks * (self.cfg.outside_roi_track_boost if outside_roi else 1.0)))
                min_sup = int(math.ceil(self.cfg.min_new_object_event_support * (self.cfg.outside_roi_event_support_boost if outside_roi else 1.0)))
                min_den = self.cfg.min_new_object_density * (self.cfg.outside_roi_density_boost if outside_roi else 1.0)
                if len(r.track_ids) < min_tracks or r.event_support < min_sup or r.density < min_den or area < self.cfg.min_new_object_area or ar > self.cfg.max_new_object_aspect_ratio:
                    self.rejected_new_object_count += 1
                    if outside_roi: self.roi_outside_reject_count += 1
                    continue
                r.object_id = self.next_obj_id; self.next_obj_id += 1
                r.state = 'tentative'; r.hits = 1; r.age = 1
            r.roi_bbox = self._expand_roi(r.last_bbox)
            self.tracked_objects[r.object_id] = r
            detections.append(r)

        lost = []
        for oid in list(unmatched_prev):
            prev = self.tracked_objects[oid]
            prev.missed_count += 1
            if prev.missed_count > self.cfg.lost_tolerance_frames:
                prev.state = 'removed'
                del self.tracked_objects[oid]
                continue
            prev.state = 'lost'
            prev.predicted = True
            lost.append(prev)

        tracked_now = [o for o in detections if o.state in {'tentative','confirmed'}] + lost
        primary = [o for o in tracked_now if o.state == 'confirmed']
        self._last_primary_objects = primary
        return {
            'raw_clusters': raw,
            'tracked_objects': tracked_now,
            'primary_objects': primary,
            'predicted_count': len(lost),
            'displayed_predicted_count': len([o for o in lost if self.cfg.display_predicted_objects]),
            'primary_merge_ms': 0.0,
            'tentative_count': len([o for o in tracked_now if o.state == 'tentative']),
            'rejected_new_object_count': self.rejected_new_object_count,
            'roi_match_count': self.roi_match_count,
            'roi_outside_reject_count': self.roi_outside_reject_count,
            'direction_reverse_reuse_count': self.direction_reverse_reuse_count,
            'tight_bbox_shrink_count': self.tight_bbox_shrink_count,
        }
