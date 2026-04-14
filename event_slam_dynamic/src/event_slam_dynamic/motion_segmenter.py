"""Track motion residual based static/dynamic segmentation."""

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

from .adapters import MotionResidual, TrackState
from .background_model import BackgroundMotionEstimator, BackgroundMotionModel


class MotionSegmenter:
    """Classify tracks by residual against background dominant motion."""

    def __init__(
        self,
        bg_estimator: BackgroundMotionEstimator,
        dynamic_residual_threshold: float,
        static_residual_threshold: float,
        dynamic_confirm_count: int,
    ) -> None:
        self._bg_estimator = bg_estimator
        self.dynamic_residual_threshold = dynamic_residual_threshold
        self.static_residual_threshold = static_residual_threshold
        self.dynamic_confirm_count = dynamic_confirm_count
        self._high_residual_count: DefaultDict[int, int] = defaultdict(int)
        self._last_residuals: Dict[int, MotionResidual] = {}

    def classify_tracks(
        self, tracks: List[TrackState], bg_model: BackgroundMotionModel, current_time: float
    ) -> Tuple[List[TrackState], List[TrackState], List[TrackState], List[MotionResidual]]:
        _ = current_time
        static_tracks: List[TrackState] = []
        dynamic_tracks: List[TrackState] = []
        uncertain_tracks: List[TrackState] = []
        residuals: List[MotionResidual] = []

        for tr in tracks:
            residual = self.compute_residual(tr, bg_model)
            residuals.append(residual)
            self._last_residuals[tr.track_id] = residual

            if residual.residual >= self.dynamic_residual_threshold:
                self._high_residual_count[tr.track_id] += 1
            else:
                self._high_residual_count[tr.track_id] = max(0, self._high_residual_count[tr.track_id] - 1)

            if self._high_residual_count[tr.track_id] >= self.dynamic_confirm_count:
                tr.is_dynamic = True
                dynamic_tracks.append(tr)
                continue

            if residual.residual <= self.static_residual_threshold:
                tr.is_dynamic = False
                static_tracks.append(tr)
            else:
                tr.is_dynamic = None
                uncertain_tracks.append(tr)

        return static_tracks, dynamic_tracks, uncertain_tracks, residuals

    def compute_residual(self, track: TrackState, bg_model: BackgroundMotionModel) -> MotionResidual:
        u, v = track.velocity_u, track.velocity_v
        u_hat, v_hat = self._bg_estimator.predict_velocity(bg_model, track.last_position[0], track.last_position[1])
        r = math.hypot(u - u_hat, v - v_hat)
        decision = "dynamic" if r >= self.dynamic_residual_threshold else "static_or_uncertain"
        return MotionResidual(
            track_id=track.track_id,
            observed_u=u,
            observed_v=v,
            predicted_u=u_hat,
            predicted_v=v_hat,
            residual=r,
            decision=decision,
        )
