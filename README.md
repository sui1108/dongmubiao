# event_slam_dynamic (minimal extensible baseline)

## 新增能力
- `result_visualizer_node.py`：订阅前端输出，OpenCV 实时叠加可视化 + 运行结果统一保存。
- `RunManager/ResultWriter/RunningSummary`：自动创建 run 目录、写入 JSONL 和 summary。
- `clusterer.py`：轻量空间+速度聚类（比一轨迹一簇更合理）。
- `debug_vis.py`：保留轨迹 marker，新增 cluster center/bbox marker。
- `dynamic_filter_with_visualization.launch`：一条命令启动前端+可视化，可选启动 RAW bridge。

## 运行方式
### 1) 手动 bridge（兼容你原来的工具）
```bash
roslaunch event_slam_dynamic dynamic_filter_with_visualization.launch start_raw_bridge:=false
python3 tools/raw_to_ros_string.py --raw /path/to/file.raw --topic /events --delta-t-us 5000 --replay-factor 1.0 --max-duration-us -1
```

### 2) 由 launch 一并启动 bridge
```bash
roslaunch event_slam_dynamic dynamic_filter_with_visualization.launch \
  start_raw_bridge:=true \
  bridge_raw_path:=/path/to/file.raw \
  bridge_delta_t_us:=5000 \
  bridge_replay_factor:=1.0 \
  bridge_max_duration_us:=-1
```

## 结果目录
默认输出到 `~/event_slam_runs/<run_id>/`，包含：
- `config.json`
- `session_info.json`（含 raw bridge 参数）
- `runtime.log`
- `static_tracks.jsonl`
- `dynamic_tracks.jsonl`
- `debug_stream.jsonl`
- `clusters.jsonl`
- `summary.json`
- `snapshots/`

可用 `python3 tools/inspect_run.py <run_dir>` 快速查看概要。

## 当前最小实现的限制
- `dynamic_filter_node.py` 当前是最小可运行示例，算法仍需替换成你的真实前端。
- clusterer 是轻量图聚类，未做时序融合与目标 ID 追踪。
- 仅实现截图保存，未默认开启视频编码（后续可新增 `export_run_video.py`）。


## RAW 自动评估
```bash
python3 tools/evaluate_raw_detection.py --check-raw-only
python3 tools/evaluate_raw_detection.py   --raw-url https://github.com/sui1108/dongmubiao/releases/download/v0.1-data/recording_2026-04-17_18-58-20.raw   --delta-t-us 10000   --show-window false   --save-every-window true
```

评估输出：
- `~/event_slam_runs/<run_id>/all_detection_frames/`
- `~/event_slam_runs/<run_id>/evaluation_report.json`
- `~/event_slam_runs/<run_id>/evaluation_report.txt`
