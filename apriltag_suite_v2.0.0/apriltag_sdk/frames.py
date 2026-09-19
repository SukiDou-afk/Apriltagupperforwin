from __future__ import annotations
import threading
import time
import numpy as np

class LatestFrameBuffer:
    """Thread-safe single-slot buffer.

    The capture thread only replaces the newest frame.  Preview and detection
    consumers pull that newest snapshot when they are ready, so old frames are
    never queued and latency cannot grow simply because a consumer is slower
    than the camera.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._timestamp = 0.0
        self._camera_matrix = None
        self._dist_coeffs = None
        self._stream_meta = {}

    def clear(self):
        with self._lock:
            self._frame = None
            self._seq = 0
            self._timestamp = 0.0
            self._camera_matrix = None
            self._dist_coeffs = None
            self._stream_meta = {}

    def set_stream_info(self, camera_matrix, dist_coeffs, **meta):
        with self._lock:
            self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
            self._dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).copy()
            self._stream_meta = dict(meta)

    def publish(self, frame, timestamp=None, **frame_meta):
        # The caller gives us an owned ndarray and never mutates it again.
        with self._lock:
            self._seq += 1
            self._stream_meta.update(frame_meta)
            self._frame = frame
            self._timestamp = float(timestamp if timestamp is not None else time.monotonic())
            return self._seq

    def snapshot(self):
        with self._lock:
            return (
                self._seq,
                self._timestamp,
                self._frame,
                self._camera_matrix,
                self._dist_coeffs,
                dict(self._stream_meta),
            )
