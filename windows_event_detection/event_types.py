from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Corner:
    corner_id: int
    x: float
    y: float
    t: int
    polarity: int
    score: float


@dataclass
class TrackPoint:
    x: float
    y: float
    t: int
    polarity: int


@dataclass
class Track:
    track_id: int
    points: List[TrackPoint] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    missed: int = 0
    confidence: float = 0.0
    residual: float = 0.0
    state: str = "uncertain"
    dyn_count: int = 0
    sta_count: int = 0


@dataclass
class DynamicObject:
    object_id: int
    last_bbox: Tuple[int, int, int, int]
    last_center: Tuple[float, float]
    velocity: Tuple[float, float]
    missed_count: int
    confidence: float
    predicted: bool = False
    track_ids: List[int] = field(default_factory=list)
    speed_mean: float = 0.0
    speed_std: float = 0.0
    direction_std: float = 0.0
    motion_consistency_score: float = 0.0
    track_count: int = 0
    event_support_count: int = 0
    residual_mean: float = 0.0
    avg_track_length: float = 0.0
