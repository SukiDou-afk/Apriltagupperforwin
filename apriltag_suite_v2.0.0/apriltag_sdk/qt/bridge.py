"""Poll latest state on the Qt thread; no per-camera-frame Qt event backlog."""
from __future__ import annotations
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class QtSystemBridge(QObject):
    frame_ready = pyqtSignal(object)
    result_ready = pyqtSignal(object)
    rgb_options = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, system, parent=None, interval_ms=33):
        super().__init__(parent)
        self.system = system
        self._frame_key = self._result_key = self._rgb_key = None
        self._status = self._error = None
        self.timer = QTimer(self)
        self.timer.setInterval(max(10, int(interval_ms)))
        self.timer.timeout.connect(self.poll)

    def start(self):
        self.timer.start()

    def stop(self):
        """Stop GUI polling only. The owner must close the AprilTagSystem."""
        self.timer.stop()

    def poll(self):
        frame = self.system.latest_frame()
        key = (frame[0], frame[1]) if frame else None
        if key != self._frame_key:
            self._frame_key = key
            self.frame_ready.emit(frame)  # None clears disconnected/stale preview.
        batch = self.system.latest_result()
        key = (batch.frame_seq, batch.captured_at) if batch else None
        if key != self._result_key:
            self._result_key = key
            self.result_ready.emit(batch)
        snapshot = self.system.camera_options_snapshot()
        if snapshot != self._rgb_key:
            self._rgb_key = snapshot
            self.rgb_options.emit(snapshot)
        state = self.system.state_snapshot()
        if state["status"] != self._status:
            self._status = state["status"]
            self.status.emit(self._status)
        if state["error"] != self._error:
            self._error = state["error"]
            if self._error:
                self.error.emit(self._error)
