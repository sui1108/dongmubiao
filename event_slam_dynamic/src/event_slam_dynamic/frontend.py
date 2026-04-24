"""Modular dynamic event frontend pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from .clusterer import DynamicTrackClusterer
from .event_corner import CornerConfig, EventCornerDetector
from .event_denoise import EventDenoiseConfig, EventDenoiser
from .event_sae import EventSAE
from .event_tracker import EventTracker, TrackerConfig
from .motion_classifier import MotionClassifier, MotionClassifierConfig


@dataclass
class FrontendConfig:
    width: int = 640
    height: int = 480


class DynamicEventFrontend:
    def __init__(self, params: Dict[str, object]):
        width = int(params.get("width", 640))
        height = int(params.get("height", 480))
        self.sae = EventSAE(width, height)
        self.denoiser = EventDenoiser(
            width,
            height,
            EventDenoiseConfig(
                refractory_us=int(params.get("refractory_us", 80)),
                support_radius=int(params.get("support_radius", 1)),
                support_dt_us=int(params.get("support_dt_us", 3000)),
                min_support_count=int(params.get("min_support_count", 1)),
                hot_pixel_count_threshold=int(params.get("hot_pixel_count_threshold", 120)),
                min_event_keep_ratio=float(params.get("min_event_keep_ratio", 0.12)),
            ),
        )
        self.corner_detector = EventCornerDetector(
            width,
            height,
            CornerConfig(
                arc_small_radius=int(params.get("arc_small_radius", 3)),
                arc_large_radius=int(params.get("arc_large_radius", 4)),
                arc_min_length=int(params.get("arc_min_length", 3)),
                arc_max_length=int(params.get("arc_max_length", 10)),
                corner_score_threshold=float(params.get("corner_score_threshold", 1500.0)),
                nms_radius=int(params.get("nms_radius", 2)),
                max_corners_per_window=int(params.get("max_corners_per_window", 300)),
                grid_rows=int(params.get("grid_rows", 6)),
                grid_cols=int(params.get("grid_cols", 8)),
                max_corners_per_cell=int(params.get("max_corners_per_cell", 12)),
            ),
        )
        self.tracker_cfg = TrackerConfig(
            match_radius_px=float(params.get("match_radius_px", 8.0)),
            match_dt_us=int(params.get("match_dt_us", 30000)),
            min_track_len=int(params.get("min_track_len", 4)),
            max_track_age=int(params.get("max_track_age", 6)),
            max_track_history=int(params.get("max_track_history", 15)),
            velocity_smooth_alpha=float(params.get("velocity_smooth_alpha", 0.6)),
        )
        self.tracker = EventTracker(self.tracker_cfg)
        self.classifier = MotionClassifier(
            MotionClassifierConfig(
                dynamic_residual_threshold=float(params.get("dynamic_residual_threshold", 0.00055)),
                static_residual_threshold=float(params.get("static_residual_threshold", 0.00025)),
                dynamic_confirm_count=int(params.get("dynamic_confirm_count", 2)),
                static_confirm_count=int(params.get("static_confirm_count", 2)),
                fallback_enable=bool(params.get("fallback_enable", True)),
                fallback_dynamic_ratio_min=float(params.get("fallback_dynamic_ratio_min", 0.05)),
                adaptive_threshold_enable=bool(params.get("adaptive_threshold_enable", True)),
            )
        )
        self.clusterer = DynamicTrackClusterer(
            spatial_threshold_px=float(params.get("cluster_spatial_threshold", 45.0)),
            min_cos_similarity=float(params.get("cluster_velocity_cos_threshold", 0.3)),
            min_tracks=int(params.get("cluster_min_tracks", 2)),
            lost_tolerance_frames=int(params.get("cluster_lost_tolerance_frames", 3)),
        )

    def process_window(self, events: List[Dict[str, int]], window_index: int, timestamp: float) -> Dict[str, object]:
        t0 = time.perf_counter()
        filtered, denoise_stats = self.denoiser.filter(events)
        corners = self.corner_detector.detect(filtered, self.sae)
        tracks = self.tracker.update(corners)
        tracks, motion_stats = self.classifier.classify(tracks, self.tracker_cfg.min_track_len)

        dynamic_tracks = [self._track_to_dict(t) for t in tracks if t.state == "dynamic"]
        static_tracks = [self._track_to_dict(t) for t in tracks if t.state == "static"]
        uncertain_tracks = [self._track_to_dict(t) for t in tracks if t.state == "uncertain"]
        clusters = [c.to_dict() for c in self.clusterer.cluster(dynamic_tracks, uncertain_tracks)]
        ms = (time.perf_counter() - t0) * 1000.0
        stats = {
            "window_event_count": len(events),
            "filtered_event_count": len(filtered),
            "corner_count": len(corners),
            "active_track_count": len(tracks),
            "dynamic_track_count": len(dynamic_tracks),
            "static_track_count": len(static_tracks),
            "uncertain_track_count": len(uncertain_tracks),
            "dynamic_cluster_count": len(clusters),
            "avg_residual": motion_stats["avg_residual"],
            "max_residual": motion_stats["max_residual"],
            "processing_ms": ms,
            "fallback_triggered": motion_stats["fallback_triggered"],
            "dynamic_threshold_used": motion_stats["dynamic_threshold_used"],
            "denoise": denoise_stats,
        }
        return {
            "timestamp": timestamp,
            "window_index": window_index,
            "events": events,
            "filtered_events": filtered,
            "corners": corners,
            "static_tracks": static_tracks,
            "dynamic_tracks": dynamic_tracks,
            "uncertain_tracks": uncertain_tracks,
            "clusters": clusters,
            "stats": stats,
        }

    @staticmethod
    def _track_to_dict(t) -> Dict[str, object]:
        return {
            "track_id": t.track_id,
            "points": [{"x": p["x"], "y": p["y"], "t": p["t"]} for p in t.points],
            "velocity": {"u": t.velocity[0], "v": t.velocity[1]},
            "age": t.age,
            "hits": t.hits,
            "missed": t.missed,
            "confidence": t.confidence,
            "residual": t.residual,
            "state": t.state,
        }
