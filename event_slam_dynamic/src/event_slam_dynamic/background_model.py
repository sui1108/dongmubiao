"""Background dominant motion estimation (minimal model).

Current fallback: global translational velocity from median reliable track velocity.
TODO: upgrade to IMU prior + RANSAC affine flow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .adapters import ImuSample, TrackState


@dataclass
class BackgroundMotionModel:
    model_type: str = "zero"
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class BackgroundMotionEstimator:
    """Estimate dominant background motion from tracks with IMU-aware fallback."""

    def estimate(
        self, tracks: List[TrackState], imu_samples: List[ImuSample], current_time: float
    ) -> BackgroundMotionModel:
        reliable = [t for t in tracks if len(t.history) >= 2]
        if len(reliable) < 2:
            # degrade gracefully with IMU-only / zero model
            omega = float(np.median([s.ang_vel_z for s in imu_samples])) if imu_samples else 0.0
            return BackgroundMotionModel(
                model_type="zero" if not imu_samples else "imu_zero_translation",
                vx=0.0,
                vy=0.0,
                omega=omega,
                confidence=0.1 if imu_samples else 0.0,
                metadata={"track_count": float(len(reliable)), "time": current_time},
            )

        us = np.array([t.velocity_u for t in reliable], dtype=np.float32)
        vs = np.array([t.velocity_v for t in reliable], dtype=np.float32)
        vx = float(np.median(us))
        vy = float(np.median(vs))
        omega = float(np.median([s.ang_vel_z for s in imu_samples])) if imu_samples else 0.0
        return BackgroundMotionModel(
            model_type="global_translation",
            vx=vx,
            vy=vy,
            omega=omega,
            confidence=min(1.0, len(reliable) / 20.0),
            metadata={"track_count": float(len(reliable)), "time": current_time},
        )

    def predict_velocity(self, model: BackgroundMotionModel, x: float, y: float) -> Tuple[float, float]:
        # placeholder: no rotation projection yet
        _ = (x, y)
        return model.vx, model.vy
