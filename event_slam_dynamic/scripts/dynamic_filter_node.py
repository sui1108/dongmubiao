#!/usr/bin/env python3
"""Dynamic front-end node with denoise/corner/track/classification pipeline."""

from __future__ import annotations

import json

import rospy
from std_msgs.msg import String

from event_slam_dynamic.frontend import DynamicEventFrontend

SCHEMA_VERSION = "1.0"


class DynamicFilterNode:
    def __init__(self) -> None:
        self.event_topic = rospy.get_param("~event_topic", "/events")
        self.static_topic = rospy.get_param("~static_topic", "/event_slam/static_tracks")
        self.dynamic_topic = rospy.get_param("~dynamic_topic", "/event_slam/dynamic_tracks")
        self.debug_topic = rospy.get_param("~debug_topic", "/event_slam/debug")

        self.frontend = DynamicEventFrontend(rospy.get_param("~", {}))
        self.pub_static = rospy.Publisher(self.static_topic, String, queue_size=5)
        self.pub_dynamic = rospy.Publisher(self.dynamic_topic, String, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, String, queue_size=5)
        self.sub_events = rospy.Subscriber(self.event_topic, String, self._on_events, queue_size=5)
        self._window_idx = 0

    def _on_events(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            events = payload.get("events", []) if isinstance(payload.get("events", []), list) else []
        except Exception:
            events = []

        self._window_idx += 1
        ts = rospy.Time.now().to_sec()
        result = self.frontend.process_window(events, self._window_idx, ts)

        self.pub_static.publish(String(data=json.dumps(self._pack_tracks("static", ts, self._window_idx, result["static_tracks"])) ))
        self.pub_dynamic.publish(String(data=json.dumps(self._pack_tracks("dynamic", ts, self._window_idx, result["dynamic_tracks"] + result["uncertain_tracks"])) ))
        self.pub_debug.publish(String(data=json.dumps({
            "schema_version": SCHEMA_VERSION,
            "stream": "debug",
            "timestamp": ts,
            "window_index": self._window_idx,
            "stats": result["stats"],
            "clusters": result["clusters"],
        })))

    @staticmethod
    def _pack_tracks(stream: str, ts: float, window_index: int, tracks):
        return {
            "schema_version": SCHEMA_VERSION,
            "stream": stream,
            "timestamp": ts,
            "window_index": window_index,
            "tracks": tracks,
        }


if __name__ == "__main__":
    rospy.init_node("dynamic_filter_node")
    DynamicFilterNode()
    rospy.loginfo("dynamic_filter_node started")
    rospy.spin()
