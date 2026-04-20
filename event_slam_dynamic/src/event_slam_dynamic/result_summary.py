"""Streaming summary aggregator for run output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RunningSummary:
    windows: int = 0
    total_event_count: int = 0
    total_corner_count: int = 0
    total_active_tracks: int = 0
    total_dynamic_tracks: int = 0
    max_dynamic_tracks: int = 0
    total_cluster_count: int = 0
    max_cluster_count: int = 0
    snapshots: list[str] = field(default_factory=list)

    def update_from_debug(self, debug_payload: Dict[str, object]) -> None:
        stats = debug_payload.get("stats", {})
        self.windows += 1
        events = int(stats.get("window_event_count", debug_payload.get("event_count", 0)))
        corners = int(stats.get("corner_count", 0))
        active = int(stats.get("active_track_count", 0))
        dyn = int(stats.get("dynamic_track_count", 0))
        self.total_event_count += events
        self.total_corner_count += corners
        self.total_active_tracks += active
        self.total_dynamic_tracks += dyn
        self.max_dynamic_tracks = max(self.max_dynamic_tracks, dyn)

    def update_clusters(self, cluster_count: int) -> None:
        self.total_cluster_count += cluster_count
        self.max_cluster_count = max(self.max_cluster_count, cluster_count)

    def to_dict(self, duration_sec: float) -> Dict[str, object]:
        denom = max(1, self.windows)
        return {
            "duration_sec": duration_sec,
            "window_count": self.windows,
            "avg_window_event_count": self.total_event_count / denom,
            "avg_corner_count": self.total_corner_count / denom,
            "avg_active_track_count": self.total_active_tracks / denom,
            "avg_dynamic_track_count": self.total_dynamic_tracks / denom,
            "max_dynamic_track_count": self.max_dynamic_tracks,
            "avg_dynamic_cluster_count": self.total_cluster_count / denom,
            "max_dynamic_cluster_count": self.max_cluster_count,
            "snapshot_paths": self.snapshots,
        }
