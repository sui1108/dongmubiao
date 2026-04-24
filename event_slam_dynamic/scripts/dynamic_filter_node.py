#!/usr/bin/env python3
"""Minimal dynamic frontend node with normalized JSON output schema.

This keeps the original String(JSON) transport and topic names while adding
stable fields for long-term visualization/logging compatibility.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List

import rospy
from std_msgs.msg import String

SCHEMA_VERSION = "1.0"


class DynamicFilterNode:
    def __init__(self) -> None:
        self.event_topic = rospy.get_param("~event_topic", "/events")
        self.static_topic = rospy.get_param("~static_topic", "/event_slam/static_tracks")
        self.dynamic_topic = rospy.get_param("~dynamic_topic", "/event_slam/dynamic_tracks")
        self.debug_topic = rospy.get_param("~debug_topic", "/event_slam/debug")
        self.window_event_limit = int(rospy.get_param("~window_event_limit", 800))

        self.pub_static = rospy.Publisher(self.static_topic, String, queue_size=5)
        self.pub_dynamic = rospy.Publisher(self.dynamic_topic, String, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, String, queue_size=5)
        self.sub_events = rospy.Subscriber(self.event_topic, String, self._on_events, queue_size=5)

        self._window_idx = 0

    def _on_events(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            events = payload.get("events", [])
        except Exception:
            events = []

        self._window_idx += 1
        ts = rospy.Time.now().to_sec()
        sampled = events[: self.window_event_limit]
        dyn_tracks: List[Dict[str, object]] = []
        sta_tracks: List[Dict[str, object]] = []

        for i, e in enumerate(sampled[:: max(1, len(sampled) // 40 + 1)]):
            x = float(e.get("x", 0.0))
            y = float(e.get("y", 0.0))
            p = int(e.get("p", 1))
            track = {
                "track_id": i,
                "points": [{"x": x, "y": y, "t": ts}],
                "velocity": {"u": math.sin(i * 0.1), "v": math.cos(i * 0.1)},
                "state": "dynamic" if p > 0 else "static",
            }
            (dyn_tracks if p > 0 else sta_tracks).append(track)

        self.pub_static.publish(String(data=json.dumps(self._pack_track_payload("static", ts, sta_tracks))))
        self.pub_dynamic.publish(String(data=json.dumps(self._pack_track_payload("dynamic", ts, dyn_tracks))))
        self.pub_debug.publish(String(data=json.dumps(self._pack_debug_payload(ts, len(events), len(dyn_tracks), len(sta_tracks)))))

    def _pack_track_payload(self, stream: str, ts: float, tracks: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stream": stream,
            "timestamp": ts,
            "window_index": self._window_idx,
            "tracks": tracks,
        }

    def _pack_debug_payload(self, ts: float, event_count: int, dyn_count: int, sta_count: int) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stream": "debug",
            "timestamp": ts,
            "window_index": self._window_idx,
            "stats": {
                "window_event_count": event_count,
                "corner_count": int(event_count * 0.08),
                "active_track_count": dyn_count + sta_count,
                "dynamic_track_count": dyn_count,
                "static_track_count": sta_count,
            },
        }


if __name__ == "__main__":
    rospy.init_node("dynamic_filter_node")
    node = DynamicFilterNode()
    rospy.loginfo("dynamic_filter_node started")
    rospy.spin()
