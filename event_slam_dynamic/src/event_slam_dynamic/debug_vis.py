"""Optional debug visualization helpers.

Uses std_msgs/String as guaranteed fallback, with MarkerArray support if available.
"""

import json
from typing import Any, Dict, List, Sequence

from std_msgs.msg import String

from .adapters import Corner, TrackState, tracks_to_dicts

try:
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    HAS_MARKERS = True
except Exception:  # pragma: no cover
    HAS_MARKERS = False


def corners_to_debug_string(corners: Sequence[Corner]) -> String:
    payload = [
        {"id": c.id, "x": c.x, "y": c.y, "t": c.t, "score": c.score, "p": c.polarity}
        for c in corners
    ]
    return String(data=json.dumps({"corners": payload}))


def tracks_to_debug_string(static_tracks: Sequence[TrackState], dynamic_tracks: Sequence[TrackState]) -> String:
    payload: Dict[str, Any] = {
        "static_tracks": tracks_to_dicts(static_tracks),
        "dynamic_tracks": tracks_to_dicts(dynamic_tracks),
    }
    return String(data=json.dumps(payload))


def tracks_to_marker_array(tracks: Sequence[TrackState], frame_id: str, ns: str, rgb: List[float]):
    """Convert tracks to MarkerArray if visualization_msgs exists, else return None."""
    if not HAS_MARKERS:
        return None
    marker_array = MarkerArray()
    for i, tr in enumerate(tracks):
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = ns
        m.id = i
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.5
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        for x, y, _ in tr.history:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            m.points.append(p)
        marker_array.markers.append(m)
    return marker_array
