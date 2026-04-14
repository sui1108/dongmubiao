"""Thread-safe buffers for events and IMU samples."""

from collections import deque
from threading import Lock
from typing import Deque, List, Optional

from .adapters import Event, ImuSample


class EventBuffer:
    """Sliding window event buffer with basic timestamp protection."""

    def __init__(self, max_buffer_sec: float) -> None:
        self._events: Deque[Event] = deque()
        self._lock = Lock()
        self._last_t: float = 0.0
        self._max_buffer_sec = max_buffer_sec

    def push_events(self, events: List[Event]) -> int:
        with self._lock:
            inserted = 0
            for e in events:
                if e.t < self._last_t - 0.5:  # severe rollback protection
                    continue
                self._events.append(e)
                self._last_t = max(self._last_t, e.t)
                inserted += 1
            if self._events:
                self.clear_old(self._last_t - self._max_buffer_sec)
            return inserted

    def get_events_in_window(self, t_start: float, t_end: float) -> List[Event]:
        if t_end < t_start:
            return []
        with self._lock:
            return [e for e in self._events if t_start <= e.t <= t_end]

    def get_latest_window(self, window_sec: float) -> List[Event]:
        with self._lock:
            if not self._events:
                return []
            t_end = self._events[-1].t
            t_start = t_end - window_sec
            return [e for e in self._events if t_start <= e.t <= t_end]

    def clear_old(self, before_time: float) -> None:
        while self._events and self._events[0].t < before_time:
            self._events.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._events)


class ImuBuffer:
    """Sliding buffer for IMU samples."""

    def __init__(self, max_buffer_sec: float) -> None:
        self._samples: Deque[ImuSample] = deque()
        self._lock = Lock()
        self._last_t: float = 0.0
        self._max_buffer_sec = max_buffer_sec

    def push_imu(self, sample: ImuSample) -> bool:
        if sample is None:
            return False
        with self._lock:
            if sample.t < self._last_t - 0.5:
                return False
            self._samples.append(sample)
            self._last_t = max(self._last_t, sample.t)
            self.clear_old(self._last_t - self._max_buffer_sec)
            return True

    def get_imu_in_window(self, t_start: float, t_end: float) -> List[ImuSample]:
        if t_end < t_start:
            return []
        with self._lock:
            return [s for s in self._samples if t_start <= s.t <= t_end]

    def get_latest_before(self, t: float) -> Optional[ImuSample]:
        with self._lock:
            latest = None
            for s in self._samples:
                if s.t <= t:
                    latest = s
                else:
                    break
            return latest

    def clear_old(self, before_time: float) -> None:
        while self._samples and self._samples[0].t < before_time:
            self._samples.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._samples)
