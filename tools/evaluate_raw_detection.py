#!/usr/bin/env python3
"""Download/check RAW and run offline dynamic detection evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "event_slam_dynamic" / "src"))

DEFAULT_URL = "https://github.com/sui1108/dongmubiao/releases/download/v0.1-data/recording_2026-04-17_18-58-20.raw"


def resolve_abs(path_str: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path_str))).resolve()


def download_raw(url: str, path: Path, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"RAW exists, skip download: {path}")
        return
    print(f"Downloading RAW from {url} -> {path}")
    try:
        urllib.request.urlretrieve(url, str(path))
        return
    except Exception as exc:
        print(f"urllib download failed: {exc}")
    for cmd in (["curl", "-L", "--fail", "-o", str(path), url], ["wget", "-O", str(path), url]):
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception:
            continue
    raise RuntimeError("RAW download failed by urllib/curl/wget")


def check_raw(path: Path, delta_t_us: int) -> int:
    if not path.exists():
        raise FileNotFoundError(f"RAW file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"RAW size invalid (0 byte): {path}")
    print(f"RAW absolute path: {path}")
    print(f"RAW size bytes: {path.stat().st_size}")
    try:
        from metavision_core.event_io import EventsIterator
    except Exception as exc:
        raise RuntimeError(f"metavision_core import failed: {exc}")

    it = EventsIterator(input_path=str(path), delta_t=delta_t_us)
    first = next(iter(it), None)
    if first is None:
        raise RuntimeError("RAW opened but first window is empty/unavailable")
    count = int(len(first))
    print(f"First window event count: {count}")
    return count


def e_to_dict(e) -> Dict[str, int]:
    return {"x": int(e[0]), "y": int(e[1]), "t": int(e[2]), "p": int(e[3])}


def evaluate(path: Path, args) -> Path:
    from metavision_core.event_io import EventsIterator
    from event_slam_dynamic.frontend import DynamicEventFrontend
    from event_slam_dynamic.run_manager import RunManager
    from event_slam_dynamic.result_writer import ResultWriter
    from event_slam_dynamic.visualizer import VisualizationComposer
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"OpenCV(cv2) missing: {exc}")

    params = {
        "width": args.width,
        "height": args.height,
        "dynamic_residual_threshold": args.dynamic_residual_threshold,
        "fallback_dynamic_ratio_min": args.fallback_dynamic_ratio_min,
        "corner_score_threshold": args.corner_score_threshold,
    }
    frontend = DynamicEventFrontend(params)
    run_mgr = RunManager(args.run_root, input_path=str(path), note="evaluate_raw_detection", bridge_info={})
    run_dir = run_mgr.start(config={"eval_args": vars(args), "frontend_params": params})
    writer = ResultWriter(run_dir)
    composer = VisualizationComposer(args.width, args.height)
    frame_dir = run_dir / "all_detection_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "total_windows": 0,
        "windows_with_events": 0,
        "windows_with_corners": 0,
        "windows_with_dynamic_tracks": 0,
        "windows_with_dynamic_clusters": 0,
        "sum_dynamic_tracks": 0,
        "sum_dynamic_clusters": 0,
        "sum_corner_survival_ratio": 0.0,
        "sum_processing_ms": 0.0,
    }
    dynamic_presence: List[int] = []

    it = EventsIterator(input_path=str(path), delta_t=args.delta_t_us)
    for idx, evs in enumerate(it, start=1):
        if args.max_windows > 0 and idx > args.max_windows:
            break
        events = [e_to_dict(e) for e in evs]
        if args.max_duration_us > 0 and events and events[-1]["t"] - events[0]["t"] > args.max_duration_us:
            break
        result = frontend.process_window(events, idx, float(events[-1]["t"]) if events else float(idx))
        stats = result["stats"]

        metrics["total_windows"] += 1
        metrics["windows_with_events"] += 1 if len(events) > 0 else 0
        metrics["windows_with_corners"] += 1 if stats["corner_count"] > 0 else 0
        metrics["windows_with_dynamic_tracks"] += 1 if stats["dynamic_track_count"] > 0 else 0
        metrics["windows_with_dynamic_clusters"] += 1 if stats["dynamic_cluster_count"] > 0 else 0
        metrics["sum_dynamic_tracks"] += stats["dynamic_track_count"]
        metrics["sum_dynamic_clusters"] += stats["dynamic_cluster_count"]
        metrics["sum_processing_ms"] += stats["processing_ms"]
        if stats["filtered_event_count"] > 0:
            metrics["sum_corner_survival_ratio"] += stats["corner_count"] / stats["filtered_event_count"]
        dynamic_presence.append(1 if (stats["dynamic_track_count"] > 0 or stats["dynamic_cluster_count"] > 0) else 0)

        static_payload = {"schema_version": "1.0", "stream": "static", "timestamp": result["timestamp"], "window_index": idx, "tracks": result["static_tracks"]}
        dynamic_payload = {"schema_version": "1.0", "stream": "dynamic", "timestamp": result["timestamp"], "window_index": idx, "tracks": result["dynamic_tracks"] + result["uncertain_tracks"]}
        debug_payload = {"schema_version": "1.0", "stream": "debug", "timestamp": result["timestamp"], "window_index": idx, "stats": stats, "clusters": result["clusters"]}
        writer.append("static", static_payload)
        writer.append("dynamic", dynamic_payload)
        writer.append("debug", debug_payload)
        writer.append("clusters", {"schema_version": "1.0", "window_index": idx, "timestamp": result["timestamp"], "clusters": result["clusters"]})

        frame = composer.compose(events, result["static_tracks"], result["dynamic_tracks"], result["uncertain_tracks"], result["clusters"], [
            f"window_index: {idx}",
            f"window_event_count: {stats['window_event_count']}",
            f"filtered_event_count: {stats['filtered_event_count']}",
            f"corner_count: {stats['corner_count']}",
            f"active_track_count: {stats['active_track_count']}",
            f"dynamic_track_count: {stats['dynamic_track_count']}",
            f"dynamic_cluster_count: {stats['dynamic_cluster_count']}",
            f"processing_ms: {stats['processing_ms']:.2f}",
        ])
        cv2.imwrite(str(frame_dir / f"frame_{idx:06d}.png"), frame)

    writer.close()

    total = max(1, metrics["total_windows"])
    corner_windows = max(1, metrics["windows_with_corners"])
    max_gap = 0
    cur = 0
    for p in dynamic_presence:
        if p == 0:
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    vis_count = len(list(frame_dir.glob("frame_*.png")))

    report = {
        **metrics,
        "dynamic_window_ratio": (metrics["windows_with_dynamic_tracks"] + metrics["windows_with_dynamic_clusters"] - metrics["windows_with_dynamic_tracks"] * 0) / float(corner_windows),
        "max_dynamic_gap": max_gap,
        "avg_dynamic_tracks": metrics["sum_dynamic_tracks"] / float(total),
        "avg_dynamic_clusters": metrics["sum_dynamic_clusters"] / float(total),
        "visualization_saved_ratio": vis_count / float(total),
        "corner_survival_ratio": metrics["sum_corner_survival_ratio"] / float(total),
        "processing_time_ms": metrics["sum_processing_ms"] / float(total),
        "visualization_count": vis_count,
        "run_dir": str(run_dir),
    }
    passed = report["visualization_saved_ratio"] >= 0.98 and report["dynamic_window_ratio"] >= 0.25 and report["max_dynamic_gap"] <= 30 and report["avg_dynamic_tracks"] > 0.3
    report["meets_requirement"] = bool(passed)

    (run_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = [
        "事件动态检测评估结论（无标注 proxy 指标）",
        f"结论：{'满足' if passed else '暂不满足'}“动目标不要明显缺失”要求。",
        f"dynamic_window_ratio={report['dynamic_window_ratio']:.3f}, max_dynamic_gap={report['max_dynamic_gap']}, avg_dynamic_tracks={report['avg_dynamic_tracks']:.3f}, avg_dynamic_clusters={report['avg_dynamic_clusters']:.3f}",
        f"visualization_saved_ratio={report['visualization_saved_ratio']:.3f}, processing_time_ms={report['processing_time_ms']:.2f}",
        "说明：以上指标不等价于真实精度，仅用于无GT素材的自检。",
        "建议重点查看 all_detection_frames 中 dynamic_gap 前后窗口。",
    ]
    if not passed:
        txt.append("建议调参：降低 dynamic_residual_threshold、降低 corner_score_threshold、放宽 match_radius_px。")
    (run_dir / "evaluation_report.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    run_mgr.finish(summary=report)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-url", default=DEFAULT_URL)
    parser.add_argument("--raw-path", default="data/recording_2026-04-17_18-58-20.raw")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--check-raw-only", action="store_true")
    parser.add_argument("--delta-t-us", type=int, default=10000)
    parser.add_argument("--show-window", default="false")
    parser.add_argument("--save-every-window", default="true")
    parser.add_argument("--max-duration-us", type=int, default=-1)
    parser.add_argument("--max-windows", type=int, default=200)
    parser.add_argument("--run-root", default="~/event_slam_runs")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dynamic-residual-threshold", type=float, default=0.00055)
    parser.add_argument("--fallback-dynamic-ratio-min", type=float, default=0.05)
    parser.add_argument("--corner-score-threshold", type=float, default=1500.0)
    args = parser.parse_args()

    raw_path = resolve_abs(args.raw_path)
    download_raw(args.raw_url, raw_path, args.force_download)
    check_raw(raw_path, args.delta_t_us)
    if args.check_raw_only:
        return
    run_dir = evaluate(raw_path, args)
    print(f"Evaluation finished: {run_dir}")


if __name__ == "__main__":
    main()
