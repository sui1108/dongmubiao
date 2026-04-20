#!/usr/bin/env python3
"""Result visualizer + unified run result recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from event_slam_dynamic.clusterer import DynamicTrackClusterer
from event_slam_dynamic.debug_vis import build_cluster_markers, build_track_markers
from event_slam_dynamic.payload_utils import normalize_track_payload, parse_json_msg
from event_slam_dynamic.result_summary import RunningSummary
from event_slam_dynamic.result_writer import ResultWriter
from event_slam_dynamic.run_manager import RunManager
from event_slam_dynamic.visualizer import VisualizationComposer


class ResultVisualizerNode:
    def __init__(self) -> None:
        self.width = int(rospy.get_param("~width", 640))
        self.height = int(rospy.get_param("~height", 480))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.use_event_background = bool(rospy.get_param("~use_event_background", True))
        self.save_snapshots = bool(rospy.get_param("~save_snapshots", True))
        self.snapshot_interval = int(rospy.get_param("~snapshot_interval", 50))

        self.event_topic = rospy.get_param("~event_topic", "/events")
        self.static_topic = rospy.get_param("~static_topic", "/event_slam/static_tracks")
        self.dynamic_topic = rospy.get_param("~dynamic_topic", "/event_slam/dynamic_tracks")
        self.debug_topic = rospy.get_param("~debug_topic", "/event_slam/debug")

        bridge_info = {
            "name": rospy.get_param("~bridge_name", "raw_to_ros_string"),
            "enabled": bool(rospy.get_param("~bridge_enabled", False)),
            "raw_input_path": rospy.get_param("~bridge_raw_path", ""),
            "topic": rospy.get_param("~bridge_topic", self.event_topic),
            "delta_t_us": int(rospy.get_param("~bridge_delta_t_us", 5000)),
            "replay_factor": float(rospy.get_param("~bridge_replay_factor", 1.0)),
            "max_duration_us": int(rospy.get_param("~bridge_max_duration_us", -1)),
        }
        self.run_manager = RunManager(
            run_root=rospy.get_param("~run_root", "~/event_slam_runs"),
            input_path=bridge_info["raw_input_path"],
            note=rospy.get_param("~run_note", ""),
            bridge_info=bridge_info,
        )
        run_dir = self.run_manager.start(config=rospy.get_param("/", {}))
        self.writer = ResultWriter(run_dir)
        self.summary = RunningSummary()

        self.composer = VisualizationComposer(self.width, self.height, show_track_ids=True)
        self.clusterer = DynamicTrackClusterer(
            spatial_threshold_px=float(rospy.get_param("~cluster_spatial_threshold", 45.0)),
            min_cos_similarity=float(rospy.get_param("~cluster_min_cos_sim", 0.3)),
        )

        self.latest_events: List[Dict[str, object]] = []
        self.latest_static: List[Dict[str, object]] = []
        self.latest_dynamic: List[Dict[str, object]] = []
        self.latest_uncertain: List[Dict[str, object]] = []
        self.latest_clusters: List[Dict[str, object]] = []
        self.latest_debug: Dict[str, object] = {}
        self.visual_frame_idx = 0

        self.sub_event = rospy.Subscriber(self.event_topic, String, self._on_events, queue_size=5)
        self.sub_static = rospy.Subscriber(self.static_topic, String, self._on_static, queue_size=5)
        self.sub_dynamic = rospy.Subscriber(self.dynamic_topic, String, self._on_dynamic, queue_size=5)
        self.sub_debug = rospy.Subscriber(self.debug_topic, String, self._on_debug, queue_size=5)
        self.pub_markers = rospy.Publisher("/event_slam/debug_markers", MarkerArray, queue_size=5)

        rospy.on_shutdown(self._on_shutdown)
        self.timer = rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~visualize_fps", 20))), self._on_timer)
        self.writer.log(f"run_id={self.run_manager.run_id} run_dir={run_dir}")

    def _on_events(self, msg: String) -> None:
        payload = parse_json_msg(msg.data)
        self.latest_events = payload.get("events", []) if isinstance(payload.get("events", []), list) else []

    def _on_static(self, msg: String) -> None:
        payload = normalize_track_payload(parse_json_msg(msg.data), "static")
        self.latest_static = payload.get("tracks", [])
        self.writer.append("static", payload)

    def _on_dynamic(self, msg: String) -> None:
        payload = normalize_track_payload(parse_json_msg(msg.data), "dynamic")
        self.latest_dynamic = payload.get("tracks", [])
        self.latest_uncertain = [t for t in self.latest_dynamic if str(t.get("state", "")) == "uncertain"]
        self.writer.append("dynamic", payload)

    def _on_debug(self, msg: String) -> None:
        payload = parse_json_msg(msg.data)
        self.latest_debug = payload
        self.summary.update_from_debug(payload)
        self.writer.append("debug", payload)

        clusters = [c.to_dict() for c in self.clusterer.cluster(self.latest_dynamic)]
        self.latest_clusters = clusters
        self.summary.update_clusters(len(clusters))

        cluster_payload = {
            "schema_version": "1.0",
            "timestamp": float(payload.get("timestamp", rospy.Time.now().to_sec())),
            "window_index": int(payload.get("window_index", self.summary.windows)),
            "clusters": clusters,
        }
        self.writer.append("clusters", cluster_payload)

    def _on_timer(self, _event: rospy.timer.TimerEvent) -> None:
        self.visual_frame_idx += 1

        marker_tracks = build_track_markers(self.latest_static, self.latest_dynamic)
        marker_clusters = build_cluster_markers(self.latest_clusters, base_id=20000)
        marker_tracks.markers.extend(marker_clusters.markers)
        self.pub_markers.publish(marker_tracks)

        stats = self.latest_debug.get("stats", {})
        lines = [
            f"run_id: {self.run_manager.run_id}",
            f"timestamp: {rospy.Time.now().to_sec():.3f}",
            f"window_events: {int(stats.get('window_event_count', len(self.latest_events)))}",
            f"corners: {int(stats.get('corner_count', 0))}",
            f"active_tracks: {int(stats.get('active_track_count', len(self.latest_static) + len(self.latest_dynamic)))}",
            f"dynamic_tracks: {len(self.latest_dynamic)}",
            f"dynamic_clusters: {len(self.latest_clusters)}",
        ]

        frame = self.composer.compose(
            events=self.latest_events,
            static_tracks=self.latest_static,
            dynamic_tracks=self.latest_dynamic,
            uncertain_tracks=self.latest_uncertain,
            clusters=self.latest_clusters,
            info_lines=lines,
            use_event_accum=self.use_event_background,
        )

        if self.save_snapshots and self.visual_frame_idx % max(1, self.snapshot_interval) == 0:
            out_path = Path(self.run_manager.run_dir) / "snapshots" / f"frame_{self.visual_frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            self.summary.snapshots.append(str(out_path))

        if self.show_window:
            cv2.imshow("event_slam_result_visualizer", frame)
            cv2.waitKey(1)

    def _on_shutdown(self) -> None:
        duration_sec = (rospy.Time.now() - rospy.Time.from_sec(self.run_manager.started_at.timestamp())).to_sec()
        summary = self.summary.to_dict(duration_sec=max(0.0, duration_sec))
        self.run_manager.finish(summary=summary)
        self.writer.log("Shutting down visualizer")
        self.writer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("result_visualizer_node")
    node = ResultVisualizerNode()
    rospy.loginfo("result_visualizer_node started")
    rospy.spin()
