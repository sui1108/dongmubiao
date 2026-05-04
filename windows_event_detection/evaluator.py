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

    id_switch = 0
    prev_ids = set()
    for s in frame_stats:
        cur = set(s.get("object_ids", []))
        if prev_ids and cur and prev_ids.isdisjoint(cur):
            id_switch += 1
        prev_ids = cur if cur else prev_ids

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
        "avg_objects_per_frame": float(sum(s["dynamic_object_count"] for s in frame_stats) / max(total, 1)),
        "avg_predicted_objects_per_frame": float(sum(s.get("predicted_object_count", 0) for s in frame_stats) / max(total, 1)),
        "same_frame_merged_count_avg": float(sum(s.get("same_frame_merged_count", 0) for s in frame_stats) / max(total, 1)),
        "matched_with_prediction_count_avg": float(sum(s.get("matched_with_prediction_count", 0) for s in frame_stats) / max(total, 1)),
        "suppressed_predicted_count_avg": float(sum(s.get("suppressed_predicted_count", 0) for s in frame_stats) / max(total, 1)),
        "final_duplicate_removed_count_avg": float(sum(s.get("final_duplicate_removed_count", 0) for s in frame_stats) / max(total, 1)),
        "id_switch_count_estimate": int(id_switch),
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
    multi_box = report["avg_dynamic_objects"] > max(1.5, report["avg_dynamic_tracks"] * 0.5)
    did_pid_overlap_risk = report["suppressed_predicted_count_avg"] < 0.1 and report["avg_predicted_objects_per_frame"] > 0.3

    txt = [
        "事件动态目标检测评估结论（无标注 proxy metrics）",
        f"1) 持续动目标: {'是' if report['dynamic_object_windows'] > 0 else '否'}",
        f"2) 明显目标缺失: {'是' if missing else '否'}",
        f"3) 是否仍存在一个目标多个框(估计): {'是' if multi_box else '否'}",
        f"4) 是否存在 D-ID 与 P-ID 同时框同一目标(风险估计): {'是' if did_pid_overlap_risk else '否'}",
        f"5) ID 是否稳定(估计): {'否' if report['id_switch_count_estimate'] > 0 else '是'}",
        "6) 建议调参: same_frame_merge_*, object_match_*, final_nms_*, bbox_*",
        "注意：无人工标注 ground truth，本报告不代表最终精度。",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation_report.txt").write_text("\n".join(txt), encoding="utf-8")
    return report
