#!/usr/bin/env python3
"""Bridge Prophesee/Metavision RAW stream to std_msgs/String JSON events.

Backwards-compatible CLI tool. If Metavision is unavailable, exits with guidance.
"""

from __future__ import annotations

import argparse
import json
import time

import rospy
from std_msgs.msg import String


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

    rospy.init_node("raw_to_ros_string", anonymous=True)
    pub = rospy.Publisher(args.topic, String, queue_size=5)

    it = EventsIterator(input_path=args.raw, delta_t=args.delta_t_us)
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
