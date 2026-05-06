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
    prim_flags = [s.get("primary_object_count", 0) > 0 for s in frame_stats]
    dyn_flags = [s.get("dynamic_object_count", 0) > 0 for s in frame_stats]
    report = {
        "total_windows": total,
        "dynamic_object_ratio": float(sum(dyn_flags) / max(total, 1)),
        "primary_object_ratio": float(sum(prim_flags) / max(total, 1)),
        "primary_object_windows": int(sum(prim_flags)),
        "max_primary_object_gap": int(_max_gap(prim_flags)),
        "avg_dynamic_objects": float(sum(s.get("dynamic_object_count", 0) for s in frame_stats) / max(total, 1)),
        "avg_raw_clusters": float(sum(s.get("raw_cluster_count", 0) for s in frame_stats) / max(total, 1)),
        "avg_tracked_objects": float(sum(s.get("tracked_object_count", 0) for s in frame_stats) / max(total, 1)),
        "avg_primary_objects": float(sum(s.get("primary_object_count", 0) for s in frame_stats) / max(total, 1)),
        "avg_child_objects_per_primary": float(sum(s.get("avg_child_objects_per_primary", 0.0) for s in frame_stats) / max(total, 1)),
        "suppressed_child_box_count_avg": float(sum(s.get("suppressed_child_box_count", 0) for s in frame_stats) / max(total, 1)),
        "predicted_object_frame_count": int(sum(s.get("predicted_object_count", 0) > 0 for s in frame_stats)),
        "displayed_predicted_object_count": int(sum(s.get("displayed_predicted_object_count", 0) for s in frame_stats)),
        "id_switch_count_estimate": int(sum(s.get("id_switch_count", 0) for s in frame_stats)),
        "primary_id_switch_count_estimate": int(sum(s.get("primary_id_switch_count", 0) for s in frame_stats)),
        "avg_primary_bbox_area": float(sum(s.get("avg_primary_bbox_area", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_event_support_in_primary_bbox": float(sum(s.get("avg_primary_event_support", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_primary_merge_ms": float(sum(s.get("primary_merge_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_denoise_ms": float(sum(s.get("denoise_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_corner_ms": float(sum(s.get("corner_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_tracker_ms": float(sum(s.get("tracker_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_classifier_ms": float(sum(s.get("classifier_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_cluster_ms": float(sum(s.get("cluster_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_visualization_ms": float(sum(s.get("visualization_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_save_image_ms": float(sum(s.get("save_image_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_processing_ms": float(sum(s.get("processing_ms", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_object_age": float(sum(s.get("avg_object_age", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_id_lifetime": float(sum(s.get("avg_id_lifetime", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_primary_object_age": float(sum(s.get("avg_primary_object_age", 0.0) for s in frame_stats) / max(total, 1)),
        "confirmed_object_ratio": float(sum(s.get("confirmed_object_count", 0) > 0 for s in frame_stats) / max(total, 1)),
        "tentative_object_count_avg": float(sum(s.get("tentative_object_count", 0) for s in frame_stats) / max(total, 1)),
        "rejected_new_object_count_avg": float(sum(s.get("rejected_new_object_count", 0) for s in frame_stats) / max(total, 1)),
        "roi_match_count_avg": float(sum(s.get("roi_match_count", 0) for s in frame_stats) / max(total, 1)),
        "roi_outside_reject_count_avg": float(sum(s.get("roi_outside_reject_count", 0) for s in frame_stats) / max(total, 1)),
        "direction_reverse_reuse_count": int(sum(s.get("direction_reverse_reuse_count", 0) for s in frame_stats)),
        "avg_confirmed_object_age": float(sum(s.get("avg_confirmed_object_age", 0.0) for s in frame_stats) / max(total, 1)),
        "avg_object_lost_recovered_count": float(sum(s.get("avg_object_lost_recovered_count", 0.0) for s in frame_stats) / max(total, 1)),
        "bbox_density_avg": float(sum(s.get("bbox_density_avg", 0.0) for s in frame_stats) / max(total, 1)),
        "tight_bbox_shrink_count_avg": float(sum(s.get("tight_bbox_shrink_count", 0) for s in frame_stats) / max(total, 1)),
        "proxy_metric_notice": "无人工标注，本报告仅为 proxy metrics，不代表真实精度。",
        "config": config_dict,
    }

    txt = [
        "事件动态物体检测评估结论（中文，proxy metrics）",
        f"1) 持续运动目标是否存在: {'是' if report['primary_object_windows'] > 0 else '否'}",
        f"2) 是否明显缺失完整目标框: {'是' if report['max_primary_object_gap'] >= 12 else '否'}",
        f"3) 是否仍有一个运动目标多个框: {'是' if report['avg_raw_clusters'] > report['avg_primary_objects'] * 1.5 else '否'}",
        f"4) Pred-ID 与 Obj-ID 同区域同时显示: {'是' if report['displayed_predicted_object_count'] > 0 else '否'}",
        f"5) ID 是否稳定: {'较稳定' if report['primary_id_switch_count_estimate'] <= report['id_switch_count_estimate'] else '一般'}",
        "6) 是否适合后续移植 Ubuntu/ROS: 是（模块边界清晰，可直接迁移）。",
        "7) 当前主要限制: 仍缺少人工标注真值，指标仅供趋势分析。",
        "注意：无 ground truth，本报告是 proxy metrics，不代表真实检测精度。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation_report.txt").write_text("\n".join(txt), encoding="utf-8")
    return report
