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


def write_evaluation(output_dir: Path, config_dict: Dict, frame_stats: List[Dict]) -> Dict:
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
        "avg_dynamic_objects": float(sum(s["dynamic_object_count"] for s in frame_stats) / max(total, 1)),
        "raw_cluster_count_avg": float(sum(s.get("raw_cluster_count", 0) for s in frame_stats) / max(total, 1)),
        "merged_object_count_avg": float(sum(s.get("merged_object_count", 0) for s in frame_stats) / max(total, 1)),
        "filtered_object_count_avg": float(sum(s.get("filtered_object_count", 0) for s in frame_stats) / max(total, 1)),
        "final_object_count_avg": float(sum(s.get("final_object_count", s.get("dynamic_object_count", 0)) for s in frame_stats) / max(total, 1)),
        "avg_motion_consistency_score": float(sum(s.get("avg_motion_consistency_score", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_speed_std": float(sum(s.get("avg_speed_std", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_direction_std": float(sum(s.get("avg_direction_std", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_cluster_time_ms": float(sum(s.get("cluster_time_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_object_merge_time_ms": float(sum(s.get("object_merge_time_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_object_tracking_time_ms": float(sum(s.get("object_tracking_time_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_final_objects": float(sum(s.get("final_object_count", 0) for s in frame_stats) / max(total, 1)),
        "avg_predicted_objects": float(sum(s.get("predicted_object_count", 0) for s in frame_stats) / max(total, 1)),
        "visualization_saved_count": int(sum(vis_flags)),
        "visualization_saved_ratio": float(sum(vis_flags) / max(total, 1)),
        "avg_processing_ms": float(sum(s["processing_ms"] for s in frame_stats) / max(total, 1)),
        "max_processing_ms": float(max((s["processing_ms"] for s in frame_stats), default=0.0)),
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
        f"5) cluster->merge->filter->final 均值: {report['raw_cluster_count_avg']:.2f} -> {report['merged_object_count_avg']:.2f} -> {report['filtered_object_count_avg']:.2f} -> {report['final_object_count_avg']:.2f}",
        f"6) 运动一致性: avg_motion_consistency_score={report['avg_motion_consistency_score']:.3f}, avg_speed_std={report['avg_speed_std']:.3f}, avg_direction_std={report['avg_direction_std']:.3f}",
        f"7) 聚类耗时(ms): cluster={report['avg_cluster_time_ms']:.3f}, merge={report['avg_object_merge_time_ms']:.3f}, tracking={report['avg_object_tracking_time_ms']:.3f}",
        f"8) 过分裂风险: {'是' if report['raw_cluster_count_avg'] > max(report['final_object_count_avg'] * 1.8, 2.0) else '否'}",
        f"9) 建议调参: cluster_velocity_cos_threshold / object_merge_velocity_cos_threshold / min_motion_consistency_score",
        "10) 适合后续移植 Ubuntu/ROS: 是（核心模块已解耦为纯 Python）。",
        "注意：无人工标注 ground truth，本报告不代表最终精度，目标缺失和过分裂判断仅为启发式。",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation_report.txt").write_text("\n".join(txt), encoding="utf-8")
    return report
