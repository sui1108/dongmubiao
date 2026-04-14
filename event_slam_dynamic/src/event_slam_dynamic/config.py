"""Configuration for dynamic filter frontend."""
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class DynamicFilterConfig:
    """Runtime configuration loaded from ROS params with safe defaults."""

    image_width: int = 640
    image_height: int = 480
    processing_rate_hz: float = 20.0
    event_window_sec: float = 0.05
    max_buffer_sec: float = 2.0

    refractory_us: float = 200.0
    isolated_dt_us: float = 3000.0
    isolated_radius: int = 2

    max_corners_per_window: int = 150
    corner_score_threshold: float = 0.3

    track_max_age_sec: float = 0.2
    track_min_length: int = 3
    track_history_max_length: int = 20
    match_distance_threshold: float = 5.0
    match_dt_threshold: float = 0.05

    dynamic_residual_threshold: float = 2.5
    static_residual_threshold: float = 1.0
    dynamic_confirm_count: int = 3

    heartbeat_interval_sec: float = 3.0
    enable_clustering: bool = True
    enable_debug_vis: bool = True

    event_topic: str = "/events"
    imu_topic: str = "/imu"
    static_tracks_topic: str = "~static_tracks"
    dynamic_tracks_topic: str = "~dynamic_tracks"
    debug_topic: str = "~debug"

    @classmethod
    def from_ros(cls) -> "DynamicFilterConfig":
        """Load config from ROS param server if rospy is available."""
        cfg = cls()
        try:
            import rospy  # pylint: disable=import-outside-toplevel

            for key, value in asdict(cfg).items():
                param_name = f"~{key}"
                setattr(cfg, key, rospy.get_param(param_name, value))
        except Exception:
            # Non-ROS contexts keep defaults.
            pass

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate config values and raise ValueError on invalid values."""
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image size must be > 0")
        if self.processing_rate_hz <= 0:
            raise ValueError("processing_rate_hz must be > 0")
        if self.event_window_sec <= 0:
            raise ValueError("event_window_sec must be > 0")
        if self.max_buffer_sec < self.event_window_sec:
            raise ValueError("max_buffer_sec must be >= event_window_sec")
        if self.dynamic_confirm_count < 1:
            raise ValueError("dynamic_confirm_count must be >= 1")
        if self.track_min_length < 2:
            raise ValueError("track_min_length must be >= 2")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debug."""
        return asdict(self)
