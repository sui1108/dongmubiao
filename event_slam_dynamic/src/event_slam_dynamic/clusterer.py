"""Lightweight clustering for dynamic tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Sequence, Tuple


@dataclass
class DynamicCluster:
    """Cluster summary built from dynamic tracks."""

    cluster_id: int
    track_ids: List[int]
    center: Tuple[float, float]
    mean_velocity: Tuple[float, float]
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "track_ids": self.track_ids,
            "track_count": len(self.track_ids),
            "center": {"x": self.center[0], "y": self.center[1]},
            "mean_velocity": {"u": self.mean_velocity[0], "v": self.mean_velocity[1]},
            "bbox": {
                "x_min": self.bbox[0],
                "y_min": self.bbox[1],
                "x_max": self.bbox[2],
                "y_max": self.bbox[3],
            },
        }


class DynamicTrackClusterer:
    """Simple spatial + velocity-consistency graph clustering.

    This avoids heavy dependencies and is intended for ROS1 runtime stability.
    """

    def __init__(self, spatial_threshold_px: float = 40.0, min_cos_similarity: float = 0.4) -> None:
        self.spatial_threshold_px = spatial_threshold_px
        self.min_cos_similarity = min_cos_similarity

    def cluster(self, dynamic_tracks: Sequence[Dict[str, object]]) -> List[DynamicCluster]:
        if not dynamic_tracks:
            return []

        neighbors: Dict[int, List[int]] = {i: [] for i in range(len(dynamic_tracks))}
        for i in range(len(dynamic_tracks)):
            for j in range(i + 1, len(dynamic_tracks)):
                if self._is_neighbor(dynamic_tracks[i], dynamic_tracks[j]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        visited = set()
        clusters: List[DynamicCluster] = []
        for i in range(len(dynamic_tracks)):
            if i in visited:
                continue
            stack = [i]
            component = []
            visited.add(i)
            while stack:
                idx = stack.pop()
                component.append(idx)
                for nxt in neighbors[idx]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            clusters.append(self._build_cluster(len(clusters), dynamic_tracks, component))

        return clusters

    def _is_neighbor(self, t1: Dict[str, object], t2: Dict[str, object]) -> bool:
        x1, y1, u1, v1 = self._extract_state(t1)
        x2, y2, u2, v2 = self._extract_state(t2)
        dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > self.spatial_threshold_px:
            return False
        dot = u1 * u2 + v1 * v2
        n1 = sqrt(u1 * u1 + v1 * v1)
        n2 = sqrt(u2 * u2 + v2 * v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return True
        cos_sim = dot / (n1 * n2)
        return cos_sim >= self.min_cos_similarity

    def _extract_state(self, track: Dict[str, object]) -> Tuple[float, float, float, float]:
        points = track.get("points", [])
        if points:
            p = points[-1]
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
        else:
            x = float(track.get("x", 0.0))
            y = float(track.get("y", 0.0))
        velocity = track.get("velocity", {})
        u = float(velocity.get("u", 0.0))
        v = float(velocity.get("v", 0.0))
        return x, y, u, v

    def _build_cluster(self, cluster_id: int, tracks: Sequence[Dict[str, object]], indices: Sequence[int]) -> DynamicCluster:
        xs: List[float] = []
        ys: List[float] = []
        us: List[float] = []
        vs: List[float] = []
        track_ids: List[int] = []
        for idx in indices:
            track = tracks[idx]
            x, y, u, v = self._extract_state(track)
            xs.append(x)
            ys.append(y)
            us.append(u)
            vs.append(v)
            track_ids.append(int(track.get("track_id", idx)))

        center = (sum(xs) / len(xs), sum(ys) / len(ys))
        mean_velocity = (sum(us) / len(us), sum(vs) / len(vs))
        bbox = (min(xs), min(ys), max(xs), max(ys))
        return DynamicCluster(
            cluster_id=cluster_id,
            track_ids=track_ids,
            center=center,
            mean_velocity=mean_velocity,
            bbox=bbox,
        )
