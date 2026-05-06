from __future__ import annotations

import argparse
from datetime import datetime
import importlib
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
from config import DetectionConfig


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows/PyCharm event RAW dynamic detection"
    )
    parser.add_argument("--raw", type=str, default=None)
    parser.add_argument("--raw-url", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--delta-t-us", type=int, default=None)
    parser.add_argument("--show-window", type=str, default=None)
    parser.add_argument("--save-every-window", type=str, default=None)
    parser.add_argument("--max-duration-us", type=int, default=None)
    parser.add_argument("--check-raw-only", action="store_true")
    return parser.parse_args()


def check_runtime_dependencies() -> bool:
    required = ["numpy", "cv2", "requests", "tqdm"]
    missing = []
    for m in required:
        try:
            importlib.import_module(m)
        except Exception as exc:  # noqa: BLE001
            missing.append((m, str(exc)))
    if missing:
        print("[error] 缺少 Python 依赖，请先安装 requirements.txt：")
        for m, err in missing:
            print(f"  - {m}: {err}")
        return False
    return True


def main() -> int:
    args = parse_args()
    cfg = DetectionConfig()
    if not check_runtime_dependencies():
        return 2

    from tqdm import tqdm
    from clusterer import DynamicObjectClusterer
    from download_data import download_raw
    from evaluator import write_evaluation
    from event_corner import EventCornerDetector
    from event_denoise import EventDenoiser
    from event_sae import EventSAE
    from event_tracker import EventTracker
    from motion_classifier import MotionClassifier
    from raw_reader import (
        RawReaderError,
        iter_event_windows,
        read_first_window_count,
        resolve_raw_path,
    )
    from visualizer import DetectionVisualizer

    if args.raw_url:
        cfg.raw_url = args.raw_url
    if args.raw:
        cfg.raw_path = Path(args.raw)
    if args.delta_t_us is not None:
        cfg.delta_t_us = args.delta_t_us
    if args.max_duration_us is not None:
        cfg.max_duration_us = args.max_duration_us
    if args.show_window is not None:
        cfg.show_window = str2bool(args.show_window)
    if args.save_every_window is not None:
        cfg.save_every_window = str2bool(args.save_every_window)

    if args.download or not cfg.raw_path.expanduser().resolve().exists():
        cfg.raw_path = download_raw(cfg.raw_url, cfg.raw_path, force_download=args.force_download)
    else:
        cfg.raw_path = resolve_raw_path(cfg.raw_path)
        print(f"[raw] RAW absolute path: {cfg.raw_path}")

    try:
        first_count, abs_raw = read_first_window_count(cfg.raw_path, cfg.delta_t_us)
        print(f"[check] EventsIterator open success: {abs_raw}")
        print(f"[check] first window event count: {first_count}")
    except RawReaderError as exc:
        print("[error] RAW check failed:")
        print(exc)
        return 2

    if args.check_raw_only:
        print("[check] --check-raw-only 完成，不执行检测流程。")
        return 0

    iterator, abs_raw, it_obj = iter_event_windows(cfg.raw_path, cfg.delta_t_us, cfg.max_duration_us)
    sensor_wh = (cfg.width, cfg.height)
    try:
        sensor_wh = it_obj.get_size() if hasattr(it_obj, "get_size") else sensor_wh
    except Exception:
        pass
    width, height = int(sensor_wh[0]), int(sensor_wh[1])
    print(f"[run] sensor size: {width}x{height}, raw={abs_raw}")

    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    out_dir = cfg.output_root / run_id
    frame_dir = out_dir / "all_detection_frames"

    denoiser = EventDenoiser(cfg, width, height)
    sae = EventSAE(width, height)
    corner_detector = EventCornerDetector(cfg, sae, width, height)
    tracker = EventTracker(cfg)
    motion = MotionClassifier(cfg)
    clusterer = DynamicObjectClusterer(cfg)
    visualizer = DetectionVisualizer(cfg, width, height, frame_dir)

    frame_stats: List[Dict[str, Any]] = []
    pbar = tqdm(desc="windows", unit="window")
    for idx, events in enumerate(iterator, start=1):
        t0 = time.perf_counter()
        filtered, denoise_stats = denoiser.filter(events)
        sae.update(filtered)
        corners = corner_detector.detect(filtered)
        frame_t = (
            int(filtered[-1]["t"])
            if len(filtered)
            else int(events[-1]["t"])
            if len(events)
            else idx * cfg.delta_t_us
        )
        tracks = tracker.update(corners, frame_t)
        st, dy, uc, mstats = motion.classify(tracks, len(filtered), len(corners))
        objects, predicted_count = clusterer.update(dy, uc, filtered)

        processing_ms = (time.perf_counter() - t0) * 1000.0
        stat = {
            "frame_idx": idx,
            "raw_count": int(len(events)),
            "filtered_count": int(len(filtered)),
            "corner_count": int(len(corners)),
            "active_track_count": int(len(tracks)),
            "dynamic_track_count": int(len(dy)),
            "dynamic_object_count": int(len(objects)),
            "processing_ms": float(processing_ms),
            "fallback_triggered": bool(mstats["fallback_triggered"]),
            "predicted_object_count": int(predicted_count),
            "id_switch_count_estimate": int(clusterer.id_switch_count_estimate),
            "bbox_shrink_suppressed_count": int(clusterer.bbox_shrink_suppressed_count),
            "event_completion_used_count": int(clusterer.event_completion_used_count),
            "avg_primary_bbox_area": float(np.mean([(o.last_bbox[2] - o.last_bbox[0]) * (o.last_bbox[3] - o.last_bbox[1]) for o in objects]) if objects else 0.0),
            "avg_bbox_area_change_ratio": float(np.mean([
                max(1.0, (o.last_bbox[2] - o.last_bbox[0]) * (o.last_bbox[3] - o.last_bbox[1])) / max(1.0, (clusterer.objects[o.object_id].last_bbox[2] - clusterer.objects[o.object_id].last_bbox[0]) * (clusterer.objects[o.object_id].last_bbox[3] - clusterer.objects[o.object_id].last_bbox[1]))
                for o in objects if o.object_id in clusterer.objects
            ]) if objects else 1.0),
            "avg_predicted_center_error": float(clusterer.total_predicted_center_error / max(1, clusterer.predicted_center_error_count)),
            "avg_event_support_in_bbox": float(np.mean([clusterer._event_support(o.last_bbox, filtered) for o in objects]) if objects else 0.0),
            "saved": False,
        }

        if cfg.save_detection_frames and cfg.save_every_window:
            visualizer.draw_and_save(idx, filtered, st, dy, uc, objects, stat)
            stat["saved"] = True

        frame_stats.append(stat)
        pbar.update(1)
    pbar.close()

    report = write_evaluation(out_dir, cfg.to_dict(), frame_stats)
    print(f"[done] output directory: {out_dir.resolve()}")
    print(
        f"[done] report dynamic_object_ratio={report['dynamic_object_ratio']:.3f}, "
        f"max_object_gap={report['max_object_gap']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
