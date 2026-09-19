from __future__ import annotations
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from ..camera import CameraCaptureWorker as CameraCore
from ..workers import DetectionWorker as DetectionCore

class CameraCaptureWorker(QThread):
    status=pyqtSignal(str)
    error=pyqtSignal(str)
    camera_info=pyqtSignal(str)
    stats=pyqtSignal(object)
    rgb_options=pyqtSignal(object)
    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.core=CameraCore(*args, **kwargs)
        for name in ("status", "error", "camera_info", "stats", "rgb_options"):
            getattr(self.core,name).connect(getattr(self,name).emit)
    def run(self):
        try:
            self.core.run()
        except Exception as exc:
            self.error.emit(str(exc))
    def stop(self):
        self.core.stop()
    def request_options(self, values):
        return self.core.request_options(values)

class DetectionWorker(QThread):
    status=pyqtSignal(str)
    error=pyqtSignal(str)
    detection_ready=pyqtSignal(object)
    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.core=DetectionCore(*args, **kwargs)
        for name in ("status", "error", "detection_ready"):
            getattr(self.core,name).connect(getattr(self,name).emit)
    def run(self):
        self.core.run()
    def stop(self):
        self.core.stop()
        self.wait(2500)
    @pyqtSlot(float)
    def set_tag_size_mm(self,v): self.core.set_tag_size_mm(v)
    @pyqtSlot(bool)
    def set_median_enabled(self,v): self.core.set_median_enabled(v)
    @pyqtSlot(float)
    def set_target_hz(self,v): self.core.set_target_hz(v)
    @pyqtSlot(object)
    def set_target_offsets_mm(self,v): self.core.set_target_offsets_mm(v)
    def _run_detection(self,*args): return self.core._run_detection(*args)
