# Windows Event Detection (Pure Python)

纯 Python（非 ROS / 非 catkin）事件相机 RAW 动态目标检测验证程序。

## 环境
- Python 3.8 / 3.9 / 3.10
- `pip install -r windows_event_detection/requirements.txt`
- 安装 Metavision SDK 并确保当前解释器可 `import metavision_core`
- 所有源码与配置文件均为标准多行文本格式（UTF-8 + LF）。

## 运行
```bash
python windows_event_detection/run_detection.py --check-raw-only --download
python windows_event_detection/run_detection.py --download --delta-t-us 10000 --show-window false --save-every-window true
```

## 常用参数
```bash
python windows_event_detection/run_detection.py --raw windows_event_detection/data/recording_2026-04-17_18-58-20.raw
python windows_event_detection/run_detection.py --raw-url <url> --download
python windows_event_detection/run_detection.py --max-duration-us 5000000
python windows_event_detection/run_detection.py --force-download --download
```

## 输出
- 可视化图像：`windows_event_detection/outputs/<run_id>/all_detection_frames/frame_000001.png`
- 评估报告：
  - `windows_event_detection/outputs/<run_id>/evaluation_report.json`
  - `windows_event_detection/outputs/<run_id>/evaluation_report.txt`

> 评估为无标注 proxy metrics，不代表最终真实精度。
