#!/usr/bin/env python3
"""ROS1 single-node dynamic target filtering frontend for event SLAM."""

import json
import traceback
from typing import Any, List, Optional

import roslib.message
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from event_slam_dynamic.adapters import Event, adapt_event_message, adapt_imu_message
from event_slam_dynamic.background_model import BackgroundMotionEstimator
from event_slam_dynamic.buffers import EventBuffer, ImuBuffer
from event_slam_dynamic.clusterer import DynamicClusterer
from event_slam_dynamic.config import DynamicFilterConfig
from event_slam_dynamic.corner_detector import CornerDetector
from event_slam_dynamic.debug_vis import HAS_MARKERS, tracks_to_marker_array
from event_slam_dynamic.motion_segmenter import MotionSegmenter
from event_slam_dynamic.preprocess import EventPreprocessor
from event_slam_dynamic.sae import TimeSurface
from event_slam_dynamic.tracker import ShortTrackTracker


class DynamicFilterNode:
    """Main ROS node orchestrating buffering, processing, and publication."""

    def __init__(self) -> None:
        rospy.init_node("dynamic_filter_node", anonymous=False)
        self.cfg = DynamicFilterConfig.from_ros()

        self.event_buffer = EventBuffer(self.cfg.max_buffer_sec)
        self.imu_buffer = ImuBuffer(self.cfg.max_buffer_sec)
        self.sae = TimeSurface(self.cfg.image_width, self.cfg.image_height, separate_polarity=True)
        self.preprocessor = EventPreprocessor(
            self.cfg.refractory_us,
            self.cfg.isolated_dt_us,
            self.cfg.isolated_radius,
        )
        self.corner_detector = CornerDetector(
            self.cfg.max_corners_per_window,
            self.cfg.corner_score_threshold,
        )
        self.tracker = ShortTrackTracker(
            track_max_age_sec=self.cfg.track_max_age_sec,
            track_min_length=self.cfg.track_min_length,
            track_history_max_length=self.cfg.track_history_max_length,
            match_distance_threshold=self.cfg.match_distance_threshold,
            match_dt_threshold=self.cfg.match_dt_threshold,
        )
        self.bg_estimator = BackgroundMotionEstimator()
        self.segmenter = MotionSegmenter(
            bg_estimator=self.bg_estimator,
            dynamic_residual_threshold=self.cfg.dynamic_residual_threshold,
            static_residual_threshold=self.cfg.static_residual_threshold,
            dynamic_confirm_count=self.cfg.dynamic_confirm_count,
        )
        self.clusterer = DynamicClusterer()

        self._last_heartbeat = rospy.Time.now().to_sec()
        self._last_window_event_count = 0
        self._last_corner_count = 0
        self._last_dynamic_count = 0
        self._last_active_track_count = 0
        self._last_processed_event_t: Optional[float] = None

        self.static_pub = rospy.Publisher(self.cfg.static_tracks_topic, String, queue_size=10)
        self.dynamic_pub = rospy.Publisher(self.cfg.dynamic_tracks_topic, String, queue_size=10)
        self.debug_pub = rospy.Publisher(self.cfg.debug_topic, String, queue_size=10)

        self.static_marker_pub = None
        self.dynamic_marker_pub = None
        if HAS_MARKERS and self.cfg.enable_debug_vis:
            from visualization_msgs.msg import MarkerArray

            self.static_marker_pub = rospy.Publisher("~static_markers", MarkerArray, queue_size=1)
            self.dynamic_marker_pub = rospy.Publisher("~dynamic_markers", MarkerArray, queue_size=1)

        self._setup_subscribers()

        rospy.Timer(rospy.Duration(1.0 / self.cfg.processing_rate_hz), self._on_process_timer)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[dynamic_filter] started with config: %s", json.dumps(self.cfg.to_dict(), ensure_ascii=False))

    def _setup_subscribers(self) -> None:
        event_msg_type = rospy.get_param("~event_msg_type", "std_msgs/String")
        event_cls = roslib.message.get_message_class(event_msg_type)
        if event_cls is None:
            rospy.logwarn(
                "[dynamic_filter] cannot resolve event_msg_type=%s. Event subscription disabled (TODO: set valid type).",
                event_msg_type,
            )
            self.event_sub = None
        else:
            self.event_sub = rospy.Subscriber(self.cfg.event_topic, event_cls, self._event_callback, queue_size=100)
        self.imu_sub = rospy.Subscriber(self.cfg.imu_topic, Imu, self._imu_callback, queue_size=200)

    def _event_callback(self, msg: Any) -> None:
        events: List[Event] = []
        if isinstance(msg, String):
            # fallback JSON schema: {"events":[{"x":..,"y":..,"t":..,"p":..}, ...]}
            try:
                payload = json.loads(msg.data)
                for e in payload.get("events", []):
                    events.append(Event(int(e["x"]), int(e["y"]), float(e["t"]), int(e.get("p", 1))))
            except Exception:
                rospy.logwarn_throttle(5.0, "[dynamic_filter] failed parsing String event payload")
        else:
            events = adapt_event_message(msg)
        if events:
            self.event_buffer.push_events(events)

    def _imu_callback(self, msg: Imu) -> None:
        sample = adapt_imu_message(msg)
        if sample is None:
            rospy.logwarn_throttle(5.0, "[dynamic_filter] IMU sample missing timestamp")
            return
        self.imu_buffer.push_imu(sample)

    def _on_process_timer(self, _event: rospy.timer.TimerEvent) -> None:
        try:
            window_events = self.event_buffer.get_latest_window(self.cfg.event_window_sec)
            self._last_window_event_count = len(window_events)
            if not window_events:
                self._heartbeat()
                return

            t_end = window_events[-1].t
            if self._last_processed_event_t is not None and t_end <= self._last_processed_event_t:
                self._heartbeat()
                return
            t_start = t_end - self.cfg.event_window_sec
            imu_samples = self.imu_buffer.get_imu_in_window(t_start, t_end)

            # processing chain
            filtered_events, pre_stats = self.preprocessor.filter_events(window_events)
            self.sae.update(filtered_events)
            corners = self.corner_detector.detect(filtered_events, self.sae)
            active_tracks = self.tracker.update(corners, t_end)
            reliable_tracks = self.tracker.get_reliable_tracks()
            bg_model = self.bg_estimator.estimate(reliable_tracks, imu_samples, t_end)
            static_tracks, dynamic_tracks, uncertain_tracks, residuals = self.segmenter.classify_tracks(
                reliable_tracks, bg_model, t_end
            )

            clusters = self.clusterer.cluster(dynamic_tracks) if self.cfg.enable_clustering else []

            self._publish_results(static_tracks, dynamic_tracks, uncertain_tracks, residuals, clusters, pre_stats.to_dict())

            self._last_corner_count = len(corners)
            self._last_dynamic_count = len(dynamic_tracks)
            self._last_active_track_count = len(active_tracks)
            self._last_processed_event_t = t_end
            self._heartbeat()
        except Exception as exc:
            rospy.logerr("[dynamic_filter] processing exception: %s", str(exc))
            rospy.logerr(traceback.format_exc())

    def _publish_results(
        self,
        static_tracks,
        dynamic_tracks,
        uncertain_tracks,
        residuals,
        clusters,
        preprocess_stats,
    ) -> None:
        static_payload = {
            "tracks": [self._track_to_simple_dict(t) for t in static_tracks],
            "count": len(static_tracks),
        }
        dynamic_payload = {
            "tracks": [self._track_to_simple_dict(t) for t in dynamic_tracks],
            "clusters": [c.__dict__ for c in clusters],
            "count": len(dynamic_tracks),
        }
        debug_payload = {
            "uncertain_count": len(uncertain_tracks),
            "residuals": [r.__dict__ for r in residuals],
            "preprocess": preprocess_stats,
        }

        self.static_pub.publish(String(data=json.dumps(static_payload)))
        self.dynamic_pub.publish(String(data=json.dumps(dynamic_payload)))
        self.debug_pub.publish(String(data=json.dumps(debug_payload)))

        if self.cfg.enable_debug_vis and HAS_MARKERS and self.static_marker_pub is not None:
            static_ma = tracks_to_marker_array(static_tracks, frame_id="event_camera", ns="static", rgb=[0.0, 1.0, 0.0])
            dynamic_ma = tracks_to_marker_array(dynamic_tracks, frame_id="event_camera", ns="dynamic", rgb=[1.0, 0.0, 0.0])
            if static_ma is not None:
                self.static_marker_pub.publish(static_ma)
            if dynamic_ma is not None:
                self.dynamic_marker_pub.publish(dynamic_ma)

    @staticmethod
    def _track_to_simple_dict(track) -> dict:
        return {
            "track_id": track.track_id,
            "x": track.last_position[0],
            "y": track.last_position[1],
            "t": track.last_timestamp,
            "u": track.velocity_u,
            "v": track.velocity_v,
            "hits": track.hits,
            "is_dynamic": track.is_dynamic,
        }

    def _heartbeat(self) -> None:
        now = rospy.Time.now().to_sec()
        if now - self._last_heartbeat < self.cfg.heartbeat_interval_sec:
            return
        self._last_heartbeat = now
        rospy.loginfo(
            "[dynamic_filter][heartbeat] buffer_events=%d window_events=%d corners=%d active_tracks=%d dynamic_tracks=%d",
            self.event_buffer.size(),
            self._last_window_event_count,
            self._last_corner_count,
            self._last_active_track_count,
            self._last_dynamic_count,
        )

    def _on_shutdown(self) -> None:
        rospy.loginfo("[dynamic_filter] shutdown requested, cleaning resources...")
        try:
            self.sae.reset()
        finally:
            rospy.loginfo("[dynamic_filter] cleanup complete")


if __name__ == "__main__":
    node: Optional[DynamicFilterNode] = None
    try:
        node = DynamicFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr("[dynamic_filter] fatal exception: %s", str(exc))
        rospy.logerr(traceback.format_exc())
    finally:
        if node is not None:
            node._on_shutdown()  # pylint: disable=protected-access
