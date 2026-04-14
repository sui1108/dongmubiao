"""Message adapters and shared dataclasses.

This module intentionally avoids hard dependency on a specific event message type.
It uses runtime introspection to support common ROS event-array layouts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Event:
    """Internal event representation.

    Attributes:
        x: Pixel x coordinate.
        y: Pixel y coordinate.
        t: Timestamp in seconds.
        p: Polarity in {-1, +1}.
    """

    x: int
    y: int
    t: float
    p: int


@dataclass
class Corner:
    """Detected corner candidate in one window."""

    id: int
    x: float
    y: float
    t: float
    score: float
    polarity: Optional[int] = None


@dataclass
class TrackState:
    """Short-term track status used by frontend segmentation."""

    track_id: int
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    last_position: Tuple[float, float] = (0.0, 0.0)
    last_timestamp: float = 0.0
    age: float = 0.0
    hits: int = 0
    velocity_u: float = 0.0
    velocity_v: float = 0.0
    confidence: float = 0.0
    is_dynamic: Optional[bool] = None


@dataclass
class MotionResidual:
    """Observed-vs-background residual result for one track."""

    track_id: int
    observed_u: float
    observed_v: float
    predicted_u: float
    predicted_v: float
    residual: float
    decision: str


@dataclass
class DynamicCluster:
    """Grouped dynamic tracks (placeholder for future contour completion)."""

    cluster_id: int
    member_track_ids: List[int]
    center_x: float
    center_y: float
    mean_u: float
    mean_v: float


@dataclass
class ImuSample:
    """Minimal IMU sample for background model usage."""

    t: float
    ang_vel_z: float = 0.0
    lin_acc_x: float = 0.0
    lin_acc_y: float = 0.0


def _stamp_to_sec(stamp: Any) -> Optional[float]:
    if stamp is None:
        return None
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    if isinstance(stamp, (float, int)):
        return float(stamp)
    return None


def _extract_pol(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else -1
    try:
        f = float(value)
        return 1 if f >= 0 else -1
    except Exception:
        return 1


def adapt_event_message(msg: Any) -> List[Event]:
    """Convert ROS event message into list of Event.

    Supported patterns (best effort):
      - msg.events[*].x/y/ts(or t)/polarity
      - msg.x/msg.y/msg.ts arrays (custom)
      - fallback: single event-like message with x,y

    TODO: Add explicit adapters when repository introduces fixed event message definitions.
    """
    base_t = _stamp_to_sec(getattr(getattr(msg, "header", None), "stamp", None))
    events: List[Event] = []

    if hasattr(msg, "events") and isinstance(msg.events, Sequence):
        for e in msg.events:
            t = _stamp_to_sec(getattr(e, "ts", None))
            if t is None:
                t = _stamp_to_sec(getattr(e, "t", None))
            if t is None:
                t = base_t
            if t is None:
                continue
            x = int(getattr(e, "x", 0))
            y = int(getattr(e, "y", 0))
            p = _extract_pol(getattr(e, "polarity", getattr(e, "p", True)))
            events.append(Event(x=x, y=y, t=t, p=p))
        return events

    if hasattr(msg, "x") and hasattr(msg, "y"):
        xs = getattr(msg, "x")
        ys = getattr(msg, "y")
        ts = getattr(msg, "ts", None)
        ps = getattr(msg, "polarity", None)
        if isinstance(xs, Iterable) and isinstance(ys, Iterable):
            xs_l = list(xs)
            ys_l = list(ys)
            ts_l = list(ts) if ts is not None else [base_t] * len(xs_l)
            ps_l = list(ps) if ps is not None else [1] * len(xs_l)
            n = min(len(xs_l), len(ys_l), len(ts_l), len(ps_l))
            for i in range(n):
                t = _stamp_to_sec(ts_l[i])
                if t is None:
                    continue
                events.append(Event(int(xs_l[i]), int(ys_l[i]), t, _extract_pol(ps_l[i])))
            return events

    # fallback single-event
    if hasattr(msg, "x") and hasattr(msg, "y") and base_t is not None:
        events.append(
            Event(
                x=int(getattr(msg, "x", 0)),
                y=int(getattr(msg, "y", 0)),
                t=base_t,
                p=_extract_pol(getattr(msg, "polarity", getattr(msg, "p", True))),
            )
        )
    return events


def adapt_imu_message(msg: Any) -> Optional[ImuSample]:
    """Convert sensor_msgs/Imu-like message to ImuSample."""
    t = _stamp_to_sec(getattr(getattr(msg, "header", None), "stamp", None))
    if t is None:
        return None
    ang = getattr(msg, "angular_velocity", None)
    acc = getattr(msg, "linear_acceleration", None)
    return ImuSample(
        t=t,
        ang_vel_z=float(getattr(ang, "z", 0.0)) if ang is not None else 0.0,
        lin_acc_x=float(getattr(acc, "x", 0.0)) if acc is not None else 0.0,
        lin_acc_y=float(getattr(acc, "y", 0.0)) if acc is not None else 0.0,
    )


def tracks_to_dicts(tracks: Sequence[TrackState]) -> List[Dict[str, Any]]:
    """Serialize tracks to dictionaries for debug publishing."""
    return [
        {
            "track_id": t.track_id,
            "last_position": {"x": t.last_position[0], "y": t.last_position[1]},
            "last_timestamp": t.last_timestamp,
            "velocity": {"u": t.velocity_u, "v": t.velocity_v},
            "hits": t.hits,
            "confidence": t.confidence,
            "is_dynamic": t.is_dynamic,
        }
        for t in tracks
    ]
