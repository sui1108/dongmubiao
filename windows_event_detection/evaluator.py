from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json


def _max_gap(flags: List[bool]) -> int:
    best = cur = 0
    for f in flags:
        if f:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def write_evaluation(output_dir: Path, config_dict: Dict, frame_stats: List[Dict], actual_width: int, actual_height: int) -> Dict:
    total = len(frame_stats)
    event_flags = [s["raw_count"] > 0 for s in frame_stats]
    corner_flags = [s["corner_count"] > 0 for s in frame_stats]
    dyn_track_flags = [s["dynamic_track_count"] > 0 for s in frame_stats]
    dyn_obj_flags = [s["dynamic_object_count"] > 0 for s in frame_stats]
    vis_flags = [s["saved"] for s in frame_stats]

    report = {
        "total_windows": total,
        "event_windows": int(sum(event_flags)),
        "corner_windows": int(sum(corner_flags)),
        "dynamic_track_windows": int(sum(dyn_track_flags)),
        "dynamic_object_windows": int(sum(dyn_obj_flags)),
        "dynamic_window_ratio": float(sum(dyn_track_flags) / max(total, 1)),
        "dynamic_object_ratio": float(sum(dyn_obj_flags) / max(total, 1)),
        "max_dynamic_gap": int(_max_gap(dyn_track_flags)),
        "max_object_gap": int(_max_gap(dyn_obj_flags)),
        "avg_dynamic_tracks": float(sum(s["dynamic_track_count"] for s in frame_stats) / max(total, 1)),
        "avg_dynamic_tracks_before_limit": float(sum(s.get("dynamic_track_count_before_limit", s["dynamic_track_count"]) for s in frame_stats) / max(total, 1)),
        "avg_dynamic_tracks_after_limit": float(sum(s["dynamic_track_count"] for s in frame_stats) / max(total, 1)),
        "avg_active_tracks": float(sum(s["active_track_count"] for s in frame_stats) / max(total, 1)),
        "avg_dynamic_objects": float(sum(s["dynamic_object_count"] for s in frame_stats) / max(total, 1)),
        "visualization_saved_count": int(sum(vis_flags)),
        "visualization_saved_ratio": float(sum(vis_flags) / max(total, 1)),
        "avg_processing_ms": float(sum(s["processing_ms"] for s in frame_stats) / max(total, 1)),
        "max_processing_ms": float(max((s["processing_ms"] for s in frame_stats), default=0.0)),
        "avg_cluster_ms": float(sum(s.get("cluster_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_visualization_ms": float(sum(s.get("visualization_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_save_image_ms": float(sum(s.get("save_image_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "id_switch_count_estimate": int(sum(1 for s in frame_stats if s.get("id_switch_estimate", 0) > 0)),
        "avg_object_age": float(sum(s.get("avg_object_age", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_id_lifetime": float(sum(s.get("avg_id_lifetime", 0.0) for s in frame_stats) / max(total, 1)),
        "actual_sensor_width": int(actual_width),
        "actual_sensor_height": int(actual_height),
        "config_width": int(config_dict.get("width", 0)),
        "config_height": int(config_dict.get("height", 0)),
        "fallback_trigger_count": int(sum(1 for s in frame_stats if s.get("fallback_triggered", False))),
        "predicted_object_frame_count": int(sum(s.get("predicted_object_count", 0) > 0 for s in frame_stats)),
        "proxy_metric_notice": "无人工标注，本报告仅为无标注 proxy metrics，不代表最终真实精度。",
        "config": config_dict,
    }

    missing = report["max_object_gap"] >= 12 or report["dynamic_object_ratio"] < 0.1
    gaps = [i + 1 for i, f in enumerate(dyn_obj_flags) if not f]
    gap_preview = gaps[:20]

    txt = [
        "事件动态目标检测评估结论（无标注 proxy metrics）",
        f"1) 持续动目标: {'是' if report['dynamic_object_windows'] > 0 else '否'}",
        f"2) 明显目标缺失: {'是' if missing else '否'}",
        f"3) 目标缺失窗口(示例): {gap_preview if gap_preview else '无'}",
        f"4) 可视化是否每窗保存: {'是' if report['visualization_saved_ratio'] > 0.99 else '否'}",
        "5) 建议调参: dynamic_residual_threshold, corner_score_threshold, cluster_min_tracks, object_lost_tolerance_frames",
        "6) 适合后续移植 Ubuntu/ROS: 是（核心模块已解耦为纯 Python）。",
        "注意：无人工标注 ground truth，本报告不代表最终精度。",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation_report.txt").write_text("\n".join(txt), encoding="utf-8")
    return report
