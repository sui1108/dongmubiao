"""Short-term nearest-neighbor corner tracker."""

import math
from typing import Dict, List, Optional

from .adapters import Corner, TrackState


class ShortTrackTracker:
    """Simple tracker for robust short tracks (not global long-term ID)."""

    def __init__(
        self,
        track_max_age_sec: float,
        track_min_length: int,
        track_history_max_length: int,
        match_distance_threshold: float,
        match_dt_threshold: float,
    ) -> None:
        self.track_max_age_sec = track_max_age_sec
        self.track_min_length = track_min_length
        self.track_history_max_length = track_history_max_length
        self.match_distance_threshold = match_distance_threshold
        self.match_dt_threshold = match_dt_threshold

        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1
        self._last_update_time: Optional[float] = None

    def update(self, corners: List[Corner], current_time: float) -> List[TrackState]:
        if self._last_update_time is not None and current_time <= self._last_update_time:
            return self.get_active_tracks()

        unmatched_track_ids = set(self._tracks.keys())
        for c in corners:
            best_id = self._find_best_track(c)
            if best_id is None:
                self._create_track(c)
            else:
                self._append_to_track(self._tracks[best_id], c)
                unmatched_track_ids.discard(best_id)

        self.prune_stale_tracks(current_time)
        self._last_update_time = current_time
        return self.get_active_tracks()

    def _find_best_track(self, corner: Corner) -> Optional[int]:
        best_id = None
        best_dist = float("inf")
        for track_id, track in self._tracks.items():
            dt = corner.t - track.last_timestamp
            if dt <= 0.0 or dt > self.match_dt_threshold:
                continue
            dx = corner.x - track.last_position[0]
            dy = corner.y - track.last_position[1]
            dist = math.hypot(dx, dy)
            if dist <= self.match_distance_threshold and dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _create_track(self, corner: Corner) -> None:
        track = TrackState(
            track_id=self._next_track_id,
            history=[(corner.x, corner.y, corner.t)],
            last_position=(corner.x, corner.y),
            last_timestamp=corner.t,
            age=0.0,
            hits=1,
            confidence=0.2,
        )
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1

    def _append_to_track(self, track: TrackState, corner: Corner) -> None:
        track.history.append((corner.x, corner.y, corner.t))
        if len(track.history) > self.track_history_max_length:
            track.history = track.history[-self.track_history_max_length :]
        track.last_position = (corner.x, corner.y)
        track.last_timestamp = corner.t
        track.hits += 1
        track.age = track.history[-1][2] - track.history[0][2]
        self.estimate_velocity(track)
        track.confidence = min(1.0, 0.1 * track.hits)

    def get_active_tracks(self) -> List[TrackState]:
        return list(self._tracks.values())

    def prune_stale_tracks(self, current_time: float) -> None:
        stale_ids = [
            tid for tid, tr in self._tracks.items() if (current_time - tr.last_timestamp) > self.track_max_age_sec
        ]
        for tid in stale_ids:
            del self._tracks[tid]

    def estimate_velocity(self, track: TrackState) -> None:
        if len(track.history) < 2:
            track.velocity_u, track.velocity_v = 0.0, 0.0
            return
        x0, y0, t0 = track.history[-2]
        x1, y1, t1 = track.history[-1]
        dt = max(1e-6, t1 - t0)
        track.velocity_u = (x1 - x0) / dt
        track.velocity_v = (y1 - y0) / dt

    def get_reliable_tracks(self) -> List[TrackState]:
        return [t for t in self._tracks.values() if len(t.history) >= self.track_min_length]
