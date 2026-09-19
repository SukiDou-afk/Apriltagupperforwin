from __future__ import annotations
import threading
import time
import traceback
import numpy as np
from collections import defaultdict, deque
from .runtime import ThreadWorker, EventHook
from .detection import DetectorEngine

class DetectionWorker(ThreadWorker):
    """AprilTag consumer that always processes the newest available frame."""


    def __init__(
        self,
        frame_buffer,
        tag_size_mm=100.0,
        median_enabled=True,
        target_offsets_mm=None,
        target_hz=15.0,
        detect_max_width=960,
        parent=None,
    ):
        super().__init__(parent)
        self.detection_ready = EventHook()
        self.status = EventHook()
        self.error = EventHook()
        self._running = False
        self._buffer = frame_buffer
        self._config_lock = threading.Lock()
        self._tag_size_mm = float(tag_size_mm)
        self._median_enabled = bool(median_enabled)
        self._target_offsets_mm = {}
        self._target_hz = max(1.0, float(target_hz))
        self._detect_max_width = max(320, int(detect_max_width))
        self._history_reset_requested = False
        self._pose_history = defaultdict(lambda: deque(maxlen=5))
        self.set_target_offsets_mm(target_offsets_mm or {})

        self.engine = DetectorEngine()
        self._pose_history = self.engine._pose_history
        self._stop_requested = threading.Event()

    def set_tag_size_mm(self, value: float):
        with self._config_lock:
            self._tag_size_mm = max(1.0, float(value))
            self._history_reset_requested = True

    def set_median_enabled(self, enabled: bool):
        with self._config_lock:
            self._median_enabled = bool(enabled)
            self._history_reset_requested = True

    def set_target_hz(self, value: float):
        with self._config_lock:
            self._target_hz = max(1.0, min(30.0, float(value)))

    def set_target_offsets_mm(self, mapping):
        parsed = {}
        try:
            for key, value in dict(mapping or {}).items():
                tag_id = int(key)
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    parsed[tag_id] = arr[:3].copy()
        except Exception:
            parsed = {}
        with self._config_lock:
            self._target_offsets_mm = parsed

    def stop(self):
        self._stop_requested.set()
        self._running = False

    def _config_snapshot(self):
        with self._config_lock:
            if self._history_reset_requested:
                self._pose_history.clear()
                self._history_reset_requested = False
            return (
                float(self._tag_size_mm),
                bool(self._median_enabled),
                {int(k): np.asarray(v, dtype=np.float64).copy()
                 for k, v in self._target_offsets_mm.items()},
                float(self._target_hz),
                int(self._detect_max_width),
            )

    def run(self):
        self._running = True
        last_seq = -1
        last_start = 0.0
        stats_t0 = time.monotonic()
        stats_count = 0
        detect_hz = 0.0
        self.status.emit("AprilTag 识别线程已启动；只处理最新帧，不追赶旧帧。")
        try:
            while self._running and not self._stop_requested.is_set():
                tag_size_mm, median_enabled, offsets, target_hz, max_width = self._config_snapshot()
                period = 1.0 / max(1.0, target_hz)
                now = time.monotonic()
                wait_s = period - (now - last_start)
                if wait_s > 0.001:
                    self.msleep(max(1, min(10, int(wait_s * 1000))))
                    continue

                seq, captured_at, frame, camera_matrix, dist_coeffs, frame_meta = self._buffer.snapshot()
                if frame is None or camera_matrix is None or dist_coeffs is None or seq == last_seq:
                    self.msleep(2)
                    continue

                last_start = time.monotonic()
                t0 = time.perf_counter()
                overlays, results = self._run_detection(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    tag_size_mm,
                    median_enabled,
                    offsets,
                    max_width,
                )
                detect_ms = (time.perf_counter() - t0) * 1000.0
                done_at = time.monotonic()
                last_seq = seq
                stats_count += 1
                stats_elapsed = done_at - stats_t0
                if stats_elapsed >= 1.0:
                    detect_hz = stats_count / stats_elapsed
                    stats_count = 0
                    stats_t0 = done_at

                self.detection_ready.emit({
                    "frame_meta": frame_meta,
                    "results": results,
                    "overlays": overlays,
                    "frame_seq": int(seq),
                    "captured_at": float(captured_at),
                    "finished_at": float(done_at),
                    "detect_ms": float(detect_ms),
                    "detect_hz": float(detect_hz),
                    "source_width": int(frame.shape[1]),
                    "source_height": int(frame.shape[0]),
                })
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(f"AprilTag 识别线程异常：{exc}")
        finally:
            self._running = False
            self.status.emit("AprilTag 识别线程已停止")

    def _run_detection(self, *args):
        return self.engine.detect(*args)
