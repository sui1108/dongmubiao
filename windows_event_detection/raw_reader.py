from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple


class RawReaderError(RuntimeError):
    pass


def import_events_iterator():
    try:
        from metavision_core.event_io import EventsIterator  # type: ignore
        return EventsIterator
    except Exception as exc:  # noqa: BLE001
        msg = (
            "无法导入 metavision_core.event_io.EventsIterator。\n"
            "请确认 Windows 已安装 Metavision SDK；\n"
            "请确认 PyCharm 使用的是 Metavision SDK 对应 Python 环境；\n"
            "请确认 Python 能 import metavision_core；\n"
            "请确认 PATH / DLL 路径正确。\n"
            f"原始错误: {exc}"
        )
        raise RawReaderError(msg) from exc


def resolve_raw_path(raw_path: Path) -> Path:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RAW file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise RawReaderError(f"RAW file size is 0: {path}")
    return path


def create_iterator(raw_path: Path, delta_t_us: int, max_duration_us: int = -1):
    EventsIterator = import_events_iterator()
    abs_raw = resolve_raw_path(raw_path)
    kwargs = {"input_path": str(abs_raw), "delta_t": int(delta_t_us)}
    if max_duration_us and max_duration_us > 0:
        kwargs["max_duration"] = int(max_duration_us)
    try:
        iterator = EventsIterator(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RawReaderError(f"EventsIterator 打开 RAW 失败: {abs_raw}\n错误: {exc}") from exc
    return iterator, abs_raw


def infer_sensor_size(iterator, default_wh: Tuple[int, int]) -> Tuple[int, int]:
    for attr in ("get_size", "sensor_size"):
        if hasattr(iterator, attr):
            try:
                v = getattr(iterator, attr)()
                if isinstance(v, tuple) and len(v) == 2:
                    return int(v[0]), int(v[1])
            except Exception:
                pass
    return default_wh


def read_first_window_count(raw_path: Path, delta_t_us: int) -> Tuple[int, Path]:
    iterator, abs_path = create_iterator(raw_path, delta_t_us)
    for evs in iterator:
        return int(len(evs)), abs_path
    return 0, abs_path


def iter_event_windows(raw_path: Path, delta_t_us: int, max_duration_us: int = -1) -> Tuple[Iterator[object], Path, object]:
    iterator, abs_path = create_iterator(raw_path, delta_t_us, max_duration_us)
    return iterator, abs_path, iterator
