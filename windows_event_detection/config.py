from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class DetectionConfig:
    raw_url: str = "https://github.com/sui1108/dongmubiao/releases/download/v0.1-data/recording_2026-04-17_18-58-20.raw"
    raw_path: Path = Path("windows_event_detection/data/recording_2026-04-17_18-58-20.raw")
    delta_t_us: int = 10_000
    max_duration_us: int = -1
    width: int = 640
    height: int = 480

    refractory_us: int = 80
    support_radius: int = 2
    support_dt_us: int = 3_000
    min_support_count: int = 2
    hot_pixel_count_threshold: int = 60
    min_event_keep_ratio: float = 0.2

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

    match_radius_px: float = 8.0
    match_dt_us: int = 30_000
    min_track_len: int = 3
    max_track_age: int = 8
    max_track_history: int = 25
    velocity_smooth_alpha: float = 0.7
    match_same_polarity: bool = False

    dynamic_residual_threshold: float = 1.7
    static_residual_threshold: float = 0.9
    dynamic_confirm_count: int = 2
    static_confirm_count: int = 2
    fallback_enable: bool = True
    adaptive_threshold_enable: bool = True

    cluster_spatial_threshold: float = 30.0
    cluster_velocity_cos_threshold: float = 0.4
    cluster_min_tracks: int = 2
    object_lost_tolerance_frames: int = 8
    object_event_support_radius: int = 20
    object_min_event_support: int = 8

    tentative_confirm_frames: int = 3
    lost_tolerance_frames: int = 8
    keep_id_on_direction_change: bool = True
    reverse_motion_tolerance: float = 0.25
    roi_expand_ratio: float = 2.0

    min_new_object_tracks: int = 3
    min_new_object_event_support: int = 12
    min_new_object_density: float = 0.01
    min_new_object_area: int = 80
    max_new_object_aspect_ratio: float = 6.0
    outside_roi_event_support_boost: float = 1.3
    outside_roi_density_boost: float = 1.3
    outside_roi_track_boost: float = 1.3

    tight_bbox_shrink_threshold: float = 0.55
    tight_bbox_padding: int = 4
    enable_tight_bbox_shrink: bool = False
    tight_bbox_shrink_density_max: float = 0.02

    primary_merge_edge_distance: float = 120.0
    primary_merge_center_distance: float = 220.0
    primary_merge_iou_threshold: float = 0.01
    primary_merge_velocity_cos_threshold: float = 0.0
    primary_merge_event_support_min: int = 20
    primary_merge_event_bridge_margin: int = 30

    primary_bbox_padding_ratio_x: float = 0.08
    primary_bbox_padding_ratio_y: float = 0.08
    primary_bbox_min_padding: int = 8
    max_primary_bbox_area_ratio: float = 0.35
    max_primary_bbox_width_ratio: float = 0.60
    max_primary_bbox_height_ratio: float = 0.85
    bbox_max_growth_ratio: float = 1.5

    predicted_max_age: int = 4
    predicted_event_support_min: int = 20
    predicted_event_support_radius: int = 30
    display_predicted_objects: bool = False
    show_lost_objects: bool = True
    show_tentative_objects: bool = False
    explicitly_count_predicted: bool = False

    show_raw_clusters: bool = False
    show_child_boxes: bool = False
    show_final_objects_only: bool = True

    min_hits_before_new_id: int = 2
    stable_age_priority: int = 5
    id_keep_bonus: float = 0.3

    max_active_tracks: int = 300
    max_dynamic_tracks: int = 300
    cluster_grid_cell_size: int = 60

    show_window: bool = False
    save_every_window: bool = True
    save_every_n_windows: int = 1
    visualization_scale: float = 1.0
    draw_track_history_len: int = 5
    draw_raw_clusters: bool = False
    save_detection_frames: bool = True
    output_root: Path = Path("windows_event_detection/outputs")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_path"] = str(self.raw_path)
        data["output_root"] = str(self.output_root)
        return data
