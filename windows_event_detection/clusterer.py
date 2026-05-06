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

    def _bbox_iou(self, a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(ua, 1e-6)

    def _event_support(self, events, bbox):
        if len(events) == 0:
            return 0
        ex = events['x'].astype(np.int32); ey = events['y'].astype(np.int32)
        return int(np.count_nonzero((ex >= bbox[0]) & (ex <= bbox[2]) & (ey >= bbox[1]) & (ey <= bbox[3])))

    def _build_raw(self, tracks: List[Track]) -> List[DynamicObject]:
        cell = self.cfg.cluster_grid_cell_size
        grid: Dict[Tuple[int, int], List[Track]] = defaultdict(list)
        for t in tracks:
            if not t.points: continue
            p = t.points[-1]
            grid[(int(p.x // cell), int(p.y // cell))].append(t)
        used, clusters = set(), []
        for t in tracks:
            if t.track_id in used or not t.points: continue
            seed = [t]; used.add(t.track_id)
            sx, sy = int(t.points[-1].x // cell), int(t.points[-1].y // cell)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for n in grid.get((sx+dx, sy+dy), []):
                        if n.track_id in used or not n.points: continue
                        d = math.hypot(t.points[-1].x-n.points[-1].x, t.points[-1].y-n.points[-1].y)
                        if d <= self.cfg.cluster_spatial_threshold:
                            seed.append(n); used.add(n.track_id)
            if len(seed) >= self.cfg.cluster_min_tracks:
                xs=[int(k.points[-1].x) for k in seed]; ys=[int(k.points[-1].y) for k in seed]
                bbox=(max(0,min(xs)-8), max(0,min(ys)-8), max(xs)+8, max(ys)+8)
                vel=np.mean(np.array([k.velocity for k in seed],dtype=np.float32), axis=0)
                clusters.append(DynamicObject(0,bbox,((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2),(float(vel[0]),float(vel[1])),0,float(np.mean([k.confidence for k in seed])),False,[k.track_id for k in seed],kind='raw_cluster'))
        return clusters

    def _within_limit(self, bbox):
        w=max(1,bbox[2]-bbox[0]); h=max(1,bbox[3]-bbox[1]); area=w*h
        return (w <= self.cfg.width * self.cfg.max_primary_bbox_width_ratio and
                h <= self.cfg.height * self.cfg.max_primary_bbox_height_ratio and
                area <= self.cfg.width * self.cfg.height * self.cfg.max_primary_bbox_area_ratio)

    def _merge_primary(self, tracked: List[DynamicObject], events: np.ndarray) -> List[DynamicObject]:
        t0=__import__('time').perf_counter()
        cell=self.cfg.cluster_grid_cell_size
        grid=defaultdict(list)
        for i,o in enumerate(tracked):
            cx,cy=o.last_center; grid[(int(cx//cell),int(cy//cell))].append(i)
        used=set(); out=[]
        for i,o in enumerate(tracked):
            if i in used: continue
            group=[i]; used.add(i)
            q=[i]
            while q:
                cur=q.pop(); co=tracked[cur]; cc=(int(co.last_center[0]//cell),int(co.last_center[1]//cell))
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        for j in grid.get((cc[0]+dx,cc[1]+dy),[]):
                            if j in used: continue
                            jo=tracked[j]
                            cd=math.hypot(co.last_center[0]-jo.last_center[0], co.last_center[1]-jo.last_center[1])
                            ed=max(0, max(co.last_bbox[0]-jo.last_bbox[2], jo.last_bbox[0]-co.last_bbox[2], co.last_bbox[1]-jo.last_bbox[3], jo.last_bbox[1]-co.last_bbox[3]))
                            iou=self._bbox_iou(co.last_bbox, jo.last_bbox)
                            ub=(min(co.last_bbox[0],jo.last_bbox[0])-self.cfg.primary_merge_event_bridge_margin,
                                min(co.last_bbox[1],jo.last_bbox[1])-self.cfg.primary_merge_event_bridge_margin,
                                max(co.last_bbox[2],jo.last_bbox[2])+self.cfg.primary_merge_event_bridge_margin,
                                max(co.last_bbox[3],jo.last_bbox[3])+self.cfg.primary_merge_event_bridge_margin)
                            es=self._event_support(events,ub)
                            if (ed<=self.cfg.primary_merge_edge_distance or cd<=self.cfg.primary_merge_center_distance or iou>=self.cfg.primary_merge_iou_threshold) and es>=self.cfg.primary_merge_event_support_min:
                                test=(min(co.last_bbox[0],jo.last_bbox[0]),min(co.last_bbox[1],jo.last_bbox[1]),max(co.last_bbox[2],jo.last_bbox[2]),max(co.last_bbox[3],jo.last_bbox[3]))
                                if self._within_limit(test):
                                    used.add(j); group.append(j); q.append(j)
            children=[tracked[k] for k in group]
            x0=min(c.last_bbox[0] for c in children); y0=min(c.last_bbox[1] for c in children)
            x1=max(c.last_bbox[2] for c in children); y1=max(c.last_bbox[3] for c in children)
            pad_x=max(self.cfg.primary_bbox_min_padding, int((x1-x0)*self.cfg.primary_bbox_padding_ratio_x))
            pad_y=max(self.cfg.primary_bbox_min_padding, int((y1-y0)*self.cfg.primary_bbox_padding_ratio_y))
            bbox=(max(0,x0-pad_x),max(0,y0-pad_y),min(self.cfg.width-1,x1+pad_x),min(self.cfg.height-1,y1+pad_y))
            if not self._within_limit(bbox): bbox=(x0,y0,x1,y1)
            trids=sorted({tid for c in children for tid in c.track_ids})
            prim=DynamicObject(children[0].object_id,bbox,((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2),children[0].velocity,0,max(c.confidence for c in children),False,trids,kind='primary_object',child_object_ids=[c.object_id for c in children],child_track_ids=trids,event_support=self._event_support(events,bbox),age=max(c.age for c in children))
            out.append(prim)
            for c in children: c.suppressed_by=prim.object_id
        self.last_primary_merge_ms=(__import__('time').perf_counter()-t0)*1000.0
        return out

    def update(self, dynamic_tracks: List[Track], uncertain_tracks: List[Track], events: np.ndarray):
        candidates=list(dynamic_tracks)
        raw=self._build_raw(candidates)
        self._last_raw_clusters=raw
        detections=[]
        for r in raw:
            best=None; bests=-1e9
            for oid,prev in self.tracked_objects.items():
                iou=self._bbox_iou(r.last_bbox, prev.last_bbox)
                cd=math.hypot(r.last_center[0]-prev.last_center[0], r.last_center[1]-prev.last_center[1])
                score=iou*2.0 - cd/max(self.cfg.primary_merge_center_distance,1.0)
                if prev.age>=self.cfg.stable_age_priority: score += self.cfg.id_keep_bonus
                if score>bests: bests=score; best=oid
            if best is not None and bests>-1.1:
                prev=self.tracked_objects[best]; r.object_id=best; r.age=prev.age+1; r.hits=prev.hits+1
            else:
                r.object_id=self.next_obj_id; self.next_obj_id+=1
            self.tracked_objects[r.object_id]=r
            if r.hits>=self.cfg.min_hits_before_new_id: detections.append(r)
        seen_ids={r.object_id for r in raw}
        predicted=[]
        for oid,prev in list(self.tracked_objects.items()):
            if oid in seen_ids:
                prev.missed_count=0; prev.predicted=False; continue
            prev.missed_count+=1
            if prev.missed_count>self.cfg.predicted_max_age:
                del self.tracked_objects[oid]; continue
            bbox=prev.last_bbox
            support=self._event_support(events,(bbox[0]-self.cfg.predicted_event_support_radius,bbox[1]-self.cfg.predicted_event_support_radius,bbox[2]+self.cfg.predicted_event_support_radius,bbox[3]+self.cfg.predicted_event_support_radius))
            if support<self.cfg.predicted_event_support_min:
                del self.tracked_objects[oid]; continue
            po=DynamicObject(prev.object_id,prev.last_bbox,prev.last_center,prev.velocity,prev.missed_count,prev.confidence,True,prev.track_ids,kind='predicted_object',age=prev.age,event_support=support)
            predicted.append(po)
        tracked_now=detections + predicted
        primary=self._merge_primary([o for o in tracked_now if not o.predicted], events)
        self._last_primary_objects=primary
        return {
            'raw_clusters': raw,
            'tracked_objects': tracked_now,
            'primary_objects': primary,
            'predicted_count': len(predicted),
            'displayed_predicted_count': 0,
            'primary_merge_ms': self.last_primary_merge_ms,
        }
