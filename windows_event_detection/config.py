from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class DetectionConfig:
    # RAW
    raw_url: str = "https://github.com/sui1108/dongmubiao/releases/download/v0.1-data/recording_2026-04-17_18-58-20.raw"
    raw_path: Path = Path("windows_event_detection/data/recording_2026-04-17_18-58-20.raw")
    delta_t_us: int = 10_000
    max_duration_us: int = -1
    width: int = 640
    height: int = 480

    # Denoise
    refractory_us: int = 80
    support_radius: int = 2
    support_dt_us: int = 3_000
    min_support_count: int = 2
    hot_pixel_count_threshold: int = 60
    min_event_keep_ratio: float = 0.2

    # Corner
    corner_method: str = "efast_arc"
    sae_recent_us: int = 6_000
    arc_small_radius: int = 3
    arc_large_radius: int = 4
    arc_min_length: int = 3
    arc_max_length: int = 10
    corner_score_threshold: float = 0.25
    nms_radius: int = 3
    max_corners_per_window: int = 800
    grid_rows: int = 6
    grid_cols: int = 8
    max_corners_per_cell: int = 25

    # Tracker
    match_radius_px: float = 8.0
    match_dt_us: int = 30_000
    min_track_len: int = 3
    max_track_age: int = 8
    max_track_history: int = 25
    velocity_smooth_alpha: float = 0.7
    match_same_polarity: bool = False

    # Motion classifier
    dynamic_residual_threshold: float = 1.7
    static_residual_threshold: float = 0.9
    dynamic_confirm_count: int = 2
    static_confirm_count: int = 2
    fallback_enable: bool = True
    adaptive_threshold_enable: bool = True

    # Object cluster / memory
    cluster_spatial_threshold: float = 30.0
    cluster_velocity_cos_threshold: float = 0.4
    cluster_min_tracks: int = 2
    object_lost_tolerance_frames: int = 8
    object_event_support_radius: int = 20
    object_min_event_support: int = 8

    # Same-frame object merge
    same_frame_merge_iou_threshold: float = 0.02
    same_frame_merge_edge_distance: float = 50.0
    same_frame_merge_center_distance: float = 120.0
    same_frame_merge_velocity_cos_threshold: float = 0.5
    same_frame_merge_speed_diff_threshold: float = 3.5

    # Cross-frame object matching
    object_match_iou_threshold: float = 0.01
    object_match_center_distance: float = 120.0
    object_match_edge_distance: float = 60.0
    object_match_velocity_cos_threshold: float = 0.45
    object_match_speed_diff_threshold: float = 4.0
    predicted_max_age: int = 8

    # BBox fusion/smoothing
    bbox_use_union_with_prediction: bool = True
    bbox_padding_ratio_x: float = 0.12
    bbox_padding_ratio_y: float = 0.12
    bbox_min_padding: int = 8
    bbox_smooth_alpha: float = 0.5

    # Final duplicate suppression
    final_nms_iou_threshold: float = 0.02
    final_nms_edge_distance: float = 40.0
    final_nms_center_distance: float = 100.0

    # Visualization
    show_window: bool = False
    save_every_window: bool = True
    save_detection_frames: bool = True
    output_root: Path = Path("windows_event_detection/outputs")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_path"] = str(self.raw_path)
        data["output_root"] = str(self.output_root)
        return data
