from __future__ import annotations

from typing import Dict, List
import math

from config import DetectionConfig
from event_types import Corner, Track, TrackPoint


class EventTracker:
    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _predict(self, tr: Track, t: int) -> tuple[float, float]:
        if not tr.points:
            return (0.0, 0.0)
        dt = max(1, t - tr.points[-1].t)
        return (tr.points[-1].x + tr.velocity[0] * dt / 1e6, tr.points[-1].y + tr.velocity[1] * dt / 1e6)

    def _update_track(self, tr: Track, c: Corner) -> None:
        if tr.points:
            prev = tr.points[-1]
            dt = max(1, c.t - prev.t)
            vx = (c.x - prev.x) * 1e6 / dt
            vy = (c.y - prev.y) * 1e6 / dt
            a = self.cfg.velocity_smooth_alpha
            tr.velocity = (a * tr.velocity[0] + (1 - a) * vx, a * tr.velocity[1] + (1 - a) * vy)
        tr.points.append(TrackPoint(c.x, c.y, c.t, c.polarity))
        tr.points = tr.points[-self.cfg.max_track_history :]
        tr.hits += 1
        tr.age += 1
        tr.missed = 0
        tr.confidence = min(1.0, tr.hits / max(1, self.cfg.min_track_len * 2))

    def update(self, corners: List[Corner], frame_t: int) -> List[Track]:
        unmatched_corners = set(range(len(corners)))
        assigned_tracks = set()

        for tid, tr in list(self.tracks.items()):
            if not corners:
                tr.missed += 1
                continue
            px, py = self._predict(tr, frame_t)
            best_idx, best_dist = -1, 1e9
            for i in list(unmatched_corners):
                c = corners[i]
                if self.cfg.match_same_polarity and tr.points and c.polarity != tr.points[-1].polarity:
                    continue
                if tr.points and c.t - tr.points[-1].t > self.cfg.match_dt_us:
                    continue
                d = math.hypot(c.x - px, c.y - py)
                if d < best_dist and d <= self.cfg.match_radius_px:
                    best_idx, best_dist = i, d
            if best_idx >= 0:
                c = corners[best_idx]
                self._update_track(tr, c)
                tr.residual = best_dist
                unmatched_corners.remove(best_idx)
                assigned_tracks.add(tid)
            else:
                tr.missed += 1
                tr.age += 1
                tr.confidence *= 0.95

        for i in unmatched_corners:
            c = corners[i]
            tr = Track(track_id=self.next_id)
            self.next_id += 1
            self._update_track(tr, c)
            self.tracks[tr.track_id] = tr

        dead = [tid for tid, tr in self.tracks.items() if tr.missed > self.cfg.max_track_age]
        for tid in dead:
            del self.tracks[tid]

        return list(self.tracks.values())
