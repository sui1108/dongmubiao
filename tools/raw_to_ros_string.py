#!/usr/bin/env python3
"""Bridge Prophesee/Metavision RAW stream to std_msgs/String JSON events."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import rospy
from std_msgs.msg import String


def resolve_raw(raw: str) -> Path:
    p = Path(os.path.abspath(os.path.expanduser(raw))).resolve()
    if not p.exists():
        raise FileNotFoundError(f"RAW not found: {p}")
    if p.stat().st_size <= 0:
        raise RuntimeError(f"RAW file empty: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Input .raw file path")
    parser.add_argument("--topic", default="/events")
    parser.add_argument("--delta-t-us", type=int, default=5000)
    parser.add_argument("--replay-factor", type=float, default=1.0)
    parser.add_argument("--max-duration-us", type=int, default=-1)
    args = parser.parse_args()

    try:
        from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
    except Exception as exc:
        raise SystemExit(f"metavision_core unavailable: {exc}")

    try:
        raw_path = resolve_raw(args.raw)
    except Exception as exc:
        raise SystemExit(f"RAW path check failed: {exc}")

    rospy.init_node("raw_to_ros_string", anonymous=True)
    pub = rospy.Publisher(args.topic, String, queue_size=5)

    it = EventsIterator(input_path=str(raw_path), delta_t=args.delta_t_us)
    replay_it = LiveReplayEventsIterator(it, replay_factor=args.replay_factor)

    start_us = None
    for evs in replay_it:
        if rospy.is_shutdown():
            break
        events = []
        for e in evs:
            t = int(e[2])
            if start_us is None:
                start_us = t
            if args.max_duration_us > 0 and t - start_us > args.max_duration_us:
                rospy.loginfo("Reached max-duration-us, stopping bridge")
                return
            events.append({"x": int(e[0]), "y": int(e[1]), "t": t, "p": int(e[3])})
        pub.publish(String(data=json.dumps({"events": events})))
        time.sleep(0.0)


if __name__ == "__main__":
    main()
