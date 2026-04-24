"""RViz marker utilities for tracks and clusters."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def _make_track_marker(track: Dict[str, object], ns: str, marker_id: int, color: tuple[float, float, float]) -> Marker:
    m = Marker()
    m.header.frame_id = "map"
    m.ns = ns
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.01
    m.color.r, m.color.g, m.color.b = color
    m.color.a = 1.0
    m.lifetime.secs = 0
    for p in track.get("points", []):
        pt = Point()
        pt.x = float(p.get("x", 0.0))
        pt.y = float(p.get("y", 0.0))
        pt.z = 0.0
        m.points.append(pt)
    return m


def build_track_markers(static_tracks: Sequence[Dict[str, object]], dynamic_tracks: Sequence[Dict[str, object]]) -> MarkerArray:
    marker_array = MarkerArray()
    marker_id = 0
    for t in static_tracks:
        marker_array.markers.append(_make_track_marker(t, "static_tracks", marker_id, (0.0, 1.0, 0.0)))
        marker_id += 1
    for t in dynamic_tracks:
        marker_array.markers.append(_make_track_marker(t, "dynamic_tracks", marker_id, (1.0, 0.0, 0.0)))
        marker_id += 1
    return marker_array


def build_cluster_markers(clusters: Iterable[Dict[str, object]], base_id: int = 10000) -> MarkerArray:
    """Render cluster bbox and center markers."""
    ma = MarkerArray()
    for i, c in enumerate(clusters):
        bbox = c.get("bbox", {})
        cx = float(c.get("center", {}).get("x", 0.0))
        cy = float(c.get("center", {}).get("y", 0.0))

        center_marker = Marker()
        center_marker.header.frame_id = "map"
        center_marker.ns = "dynamic_cluster_centers"
        center_marker.id = base_id + i * 2
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        center_marker.pose.position.x = cx
        center_marker.pose.position.y = cy
        center_marker.pose.position.z = 0.0
        center_marker.scale.x = 0.08
        center_marker.scale.y = 0.08
        center_marker.scale.z = 0.08
        center_marker.color.r = 1.0
        center_marker.color.g = 0.2
        center_marker.color.b = 0.2
        center_marker.color.a = 1.0
        ma.markers.append(center_marker)

        bbox_marker = Marker()
        bbox_marker.header.frame_id = "map"
        bbox_marker.ns = "dynamic_cluster_bbox"
        bbox_marker.id = base_id + i * 2 + 1
        bbox_marker.type = Marker.CUBE
        bbox_marker.action = Marker.ADD
        x_min = float(bbox.get("x_min", cx))
        y_min = float(bbox.get("y_min", cy))
        x_max = float(bbox.get("x_max", cx))
        y_max = float(bbox.get("y_max", cy))
        bbox_marker.pose.position.x = (x_min + x_max) / 2.0
        bbox_marker.pose.position.y = (y_min + y_max) / 2.0
        bbox_marker.pose.position.z = 0.0
        bbox_marker.scale.x = max(0.05, x_max - x_min)
        bbox_marker.scale.y = max(0.05, y_max - y_min)
        bbox_marker.scale.z = 0.01
        bbox_marker.color.r = 1.0
        bbox_marker.color.g = 0.0
        bbox_marker.color.b = 0.0
        bbox_marker.color.a = 0.2
        ma.markers.append(bbox_marker)
    return ma
