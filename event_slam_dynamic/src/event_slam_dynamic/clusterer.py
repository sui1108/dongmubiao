"""Spatial+velocity clustering with short memory."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Sequence, Tuple


@dataclass
class DynamicCluster:
    cluster_id: int
    track_ids: List[int]
    center: Tuple[float, float]
    mean_velocity: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    predicted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "member_track_ids": self.track_ids,
            "bbox": {"x_min": self.bbox[0], "y_min": self.bbox[1], "x_max": self.bbox[2], "y_max": self.bbox[3]},
            "center": {"x": self.center[0], "y": self.center[1]},
            "avg_velocity": {"u": self.mean_velocity[0], "v": self.mean_velocity[1]},
            "confidence": self.confidence,
            "predicted": self.predicted,
        }


class DynamicTrackClusterer:
    def __init__(self, spatial_threshold_px: float = 40.0, min_cos_similarity: float = 0.4, min_tracks: int = 2, lost_tolerance_frames: int = 3):
        self.spatial_threshold_px = spatial_threshold_px
        self.min_cos_similarity = min_cos_similarity
        self.min_tracks = min_tracks
        self.lost_tolerance_frames = lost_tolerance_frames
        self._memory: List[Tuple[DynamicCluster, int]] = []

    def cluster(self, dynamic_tracks: Sequence[Dict[str, object]], uncertain_tracks: Sequence[Dict[str, object]] | None = None) -> List[DynamicCluster]:
        tracks = list(dynamic_tracks)
        if len(tracks) < self.min_tracks and uncertain_tracks:
            tracks.extend([t for t in uncertain_tracks if float(t.get("residual", 0.0)) > 0.0004])
        clusters: List[DynamicCluster] = []
        if tracks:
            neighbors = {i: [] for i in range(len(tracks))}
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    if self._is_neighbor(tracks[i], tracks[j]):
                        neighbors[i].append(j)
                        neighbors[j].append(i)
            visited = set()
            for i in range(len(tracks)):
                if i in visited:
                    continue
                stack, comp = [i], []
                visited.add(i)
                while stack:
                    idx = stack.pop()
                    comp.append(idx)
                    for nxt in neighbors[idx]:
                        if nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)
                if len(comp) >= self.min_tracks:
                    clusters.append(self._build_cluster(len(clusters), tracks, comp))

        self._memory = [(c, 0) for c in clusters] + [(c, age + 1) for c, age in self._memory if age + 1 <= self.lost_tolerance_frames]
        if clusters:
            return clusters
        return [DynamicCluster(c.cluster_id, c.track_ids, c.center, c.mean_velocity, c.bbox, c.confidence * 0.5, predicted=True) for c, _ in self._memory]

    def _is_neighbor(self, t1: Dict[str, object], t2: Dict[str, object]) -> bool:
        x1, y1, u1, v1 = self._extract_state(t1)
        x2, y2, u2, v2 = self._extract_state(t2)
        dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > self.spatial_threshold_px:
            return False
        dot = u1 * u2 + v1 * v2
        n1 = sqrt(u1 * u1 + v1 * v1)
        n2 = sqrt(u2 * u2 + v2 * v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return True
        return (dot / (n1 * n2)) >= self.min_cos_similarity

    @staticmethod
    def _extract_state(track: Dict[str, object]) -> Tuple[float, float, float, float]:
        p = (track.get("points") or [{"x": track.get("x", 0.0), "y": track.get("y", 0.0)}])[-1]
        v = track.get("velocity", {})
        return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(v.get("u", 0.0)), float(v.get("v", 0.0))

    def _build_cluster(self, cid: int, tracks: Sequence[Dict[str, object]], indices: Sequence[int]) -> DynamicCluster:
        xs, ys, us, vs, ids = [], [], [], [], []
        for i in indices:
            x, y, u, v = self._extract_state(tracks[i])
            xs.append(x); ys.append(y); us.append(u); vs.append(v)
            ids.append(int(tracks[i].get("track_id", i)))
        return DynamicCluster(
            cluster_id=cid,
            track_ids=ids,
            center=(sum(xs) / len(xs), sum(ys) / len(ys)),
            mean_velocity=(sum(us) / len(us), sum(vs) / len(vs)),
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            confidence=min(1.0, len(indices) / 6.0),
        )
