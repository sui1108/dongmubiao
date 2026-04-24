#!/usr/bin/env python3
"""ROS-launch friendly wrapper that republishes RAW as String(JSON) events."""

from __future__ import annotations

import json
import os
from pathlib import Path

import rospy
from std_msgs.msg import String


def resolve_raw_path(raw_path: str) -> Path:
    expanded = os.path.expanduser(raw_path)
    abs_path = os.path.abspath(expanded)
    resolved = Path(abs_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"RAW file not found: {resolved}")
    if resolved.stat().st_size <= 0:
        raise RuntimeError(f"RAW file empty: {resolved}")
    return resolved


def main() -> None:
    rospy.init_node("raw_bridge_node")
    raw_path = rospy.get_param("~raw_path", "")
    topic = rospy.get_param("~topic", "/events")
    delta_t_us = int(rospy.get_param("~delta_t_us", 5000))
    replay_factor = float(rospy.get_param("~replay_factor", 1.0))
    max_duration_us = int(rospy.get_param("~max_duration_us", -1))

    if not raw_path:
        rospy.logerr("~raw_path is required when bridge is enabled")
        return

    try:
        from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
    except Exception as exc:
        rospy.logerr(f"metavision_core dependency missing: {exc}")
        return

    try:
        resolved = resolve_raw_path(raw_path)
    except Exception as exc:
        rospy.logerr(str(exc))
        return

    pub = rospy.Publisher(topic, String, queue_size=5)
    it = EventsIterator(input_path=str(resolved), delta_t=delta_t_us)
    replay_it = LiveReplayEventsIterator(it, replay_factor=replay_factor)
    start_us = None
    rospy.loginfo(f"RAW bridge input: {resolved}")

    for evs in replay_it:
        if rospy.is_shutdown():
            break
        events = []
        for e in evs:
            t = int(e[2])
            if start_us is None:
                start_us = t
            if max_duration_us > 0 and t - start_us > max_duration_us:
                rospy.loginfo("Reached bridge max_duration_us, stopping")
                return
            events.append({"x": int(e[0]), "y": int(e[1]), "t": t, "p": int(e[3])})
        pub.publish(String(data=json.dumps({"events": events})))


if __name__ == "__main__":
    main()
