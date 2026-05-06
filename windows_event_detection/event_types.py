from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    kind: str = "tracked_object"
    child_track_ids: List[int] = field(default_factory=list)
    child_object_ids: List[int] = field(default_factory=list)
    suppressed_by: Optional[int] = None
    age: int = 1
    hits: int = 1
    event_support: int = 0
    state: str = "tentative"
    density: float = 0.0
    bbox_density: float = 0.0
    tight_bbox_area_ratio: float = 1.0
    reused_id: bool = False
    reverse_motion: bool = False
    lost_recovered_count: int = 0
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
