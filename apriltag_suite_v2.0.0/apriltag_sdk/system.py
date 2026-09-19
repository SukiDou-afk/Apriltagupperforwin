"""Managed RealSense + latest-frame recognition facade; Qt is optional."""
from __future__ import annotations
import copy
import threading
import time
from .locator import Locator
from .frames import LatestFrameBuffer
from .gimbal import GimbalController
from .runtime import EventHook
from .config import load_settings, save_settings


class AprilTagSystem:
    """Lifecycle/configuration methods belong to one owner thread.

    EventHook callbacks execute on worker threads: keep them short and do not
    touch widgets or call start/stop inside them. QtSystemBridge polls safely.
    """
    def __init__(self, locator=None, config_path=None, detection_hz=15):
        self.locator = locator or Locator()
        self.gimbal = GimbalController()
        self.status, self.error, self.result_ready = EventHook(), EventHook(), EventHook()
        self.rgb_options = EventHook()
        self.frames = LatestFrameBuffer()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._camera, self._detector = None, None
        self._latest = None
        self._rgb_snapshot = None
        self._rgb_memory = {}
        self._restored_serials = set()
        self._latest_status = "未启动"
        self._latest_error = ""
        self.config_path = config_path
        self.detection_hz = float(detection_hz)
        if not 1 <= self.detection_hz <= 30:
            raise ValueError("detection_hz must be between 1 and 30")
        self.gimbal.error.connect(self._record_error)
        if config_path:
            self.load_config(config_path)

    @property
    def running(self):
        return bool(self._camera and self._camera.is_alive() and not self._stop.is_set())

    def state_snapshot(self):
        with self._lock:
            return dict(running=self.running, status=self._latest_status, error=self._latest_error)

    def _record_status(self, text):
        with self._lock:
            self._latest_status = str(text)
        self.status.emit(str(text))

    def _record_error(self, text):
        self._stop.set()
        with self._lock:
            self._latest_error = str(text)
            self._latest = None
        camera = self._camera
        if camera:
            camera.stop()
        self.error.emit(str(text))

    def start_camera(self, width=None, height=None, fps=None, pixel_format="bgr8", serial=None):
        """No dimensions: discover and choose highest supported colour profile.

        Explicit dimensions: all width/height/fps required, pixel_format is SDK
        enum name (bgr8/rgb8/yuyv/uyvy/mjpeg/bgra8/rgba8).
        Startup is asynchronous; inspect state/error or connect the error hook.
        """
        if any(w and w.is_alive() for w in (self._camera, self._detector)):
            raise RuntimeError("camera workers still active; call stop_camera first")
        from .camera import CameraCaptureWorker, discover_color_profiles, _rs_format
        if width is None and height is None and fps is None:
            _, discovered_serial, profiles = discover_color_profiles(serial=serial)
            if not profiles:
                raise RuntimeError("未发现可用 RealSense RGB 相机")
            width, height, fps, fmt = profiles[0]
            serial = serial or discovered_serial
        elif None in (width, height, fps):
            raise ValueError("provide all of width, height and fps, or omit all three")
        else:
            fmt = _rs_format(pixel_format)
            if fmt is None:
                raise ValueError("unknown RealSense pixel_format")
        self.frames.clear()
        self._stop.clear()
        with self._lock:
            self._latest = None
            self._rgb_snapshot = None
            self._restored_serials.clear()
            self._latest_error = ""
        camera = CameraCaptureWorker(self.frames, width, height, fps, fmt,
            serial=serial, pose_provider=self.gimbal.command_snapshot)
        camera.status.connect(self._record_status)
        camera.camera_info.connect(self._record_status)
        camera.error.connect(self._record_error)
        camera.rgb_options.connect(self._on_rgb_options)
        self._camera = camera
        self._detector = threading.Thread(target=self._detect_loop, name="apriltag-locator", daemon=True)
        camera.start()
        self._detector.start()

    def _detect_loop(self):
        last_seq = -1
        try:
            while not self._stop.is_set():
                start = time.monotonic()
                seq, stamp, image, matrix, distortion, meta = self.frames.snapshot()
                if image is not None and seq != last_seq and matrix is not None:
                    batch = self.locator.process(image, matrix, distortion,
                        meta.get("pan_cmd_deg"), meta.get("tilt_cmd_deg"), stamp, seq)
                    last_seq = seq
                    with self._lock:
                        if self._stop.is_set():
                            break
                        self._latest = batch
                    self.result_ready.emit(copy.deepcopy(batch))
                self._stop.wait(max(0.002, 1 / self.detection_hz - (time.monotonic() - start)))
        except Exception as exc:
            self._record_error("识别异常：" + str(exc))

    def latest_result(self, max_age_s=1.0):
        """Independent snapshot, or None when absent/stale. Empty targets is valid."""
        with self._lock:
            batch = self._latest
            if batch is None or (max_age_s is not None and batch.age_s > max_age_s):
                return None
            return copy.deepcopy(batch)

    def latest_frame(self, max_age_s=1.0):
        """Return (seq, monotonic timestamp, owned BGR image, K, distortion, meta)."""
        seq, stamp, image, matrix, distortion, meta = self.frames.snapshot()
        if image is None or (max_age_s is not None and time.monotonic() - stamp > max_age_s):
            return None
        return seq, stamp, image.copy(), matrix.copy(), distortion.copy(), meta

    def request_camera_options(self, values):
        if not self.running:
            raise RuntimeError("camera is not running")
        generation = self._camera.request_options(dict(values))
        if generation is None:
            raise RuntimeError("RGB options are not ready; wait for first snapshot")
        return generation

    def _on_rgb_options(self, snapshot):
        serial = str(snapshot["serial"])
        with self._lock:
            self._rgb_snapshot = copy.deepcopy(snapshot)
            self._rgb_memory.setdefault(serial, {}).update(snapshot.get("applied", {}))
            restore = None
            if serial not in self._restored_serials:
                self._restored_serials.add(serial)
                restore = dict(self._rgb_memory.get(serial, {}))
                from .camera_options import DEPENDENCIES
                for key, auto in DEPENDENCIES.items():
                    current_auto = snapshot.get("records", {}).get(auto, {}).get("value", 1)
                    if restore.get(auto, current_auto) != 0:
                        restore.pop(key, None)
        if restore:
            self._camera.request_options(restore)
        self.rgb_options.emit(copy.deepcopy(snapshot))

    def camera_options_snapshot(self):
        with self._lock:
            return copy.deepcopy(self._rgb_snapshot)

    def stop_camera(self, timeout_s=5.0):
        self._stop.set()
        if self._camera:
            self._camera.stop()
        deadline = time.monotonic() + timeout_s
        for worker in (self._detector, self._camera):
            if worker and worker.ident is not None:
                if worker is threading.current_thread():
                    raise RuntimeError("call stop_camera from the owner thread, not a callback")
                worker.join(max(0, deadline - time.monotonic()))
        if any(w and w.is_alive() for w in (self._detector, self._camera)):
            raise TimeoutError("相机工作线程尚未退出，请稍后再次 stop_camera；不要销毁所属对象")
        self._camera = self._detector = None
        self.frames.clear()
        with self._lock:
            self._latest = None
            self._rgb_snapshot = None

    def save_config(self, path=None):
        target = path or self.config_path
        if not target:
            raise ValueError("provide config path")
        with self._lock:
            memory = copy.deepcopy(self._rgb_memory)
        save_settings(target, dict(schema_version=1, locator=self.locator.export_state(),
            rgb_options=memory, home=list(self.gimbal.home_angles),
            tx_rate_hz=self.gimbal.tx_snapshot()["tx_rate_hz"], detection_hz=self.detection_hz))

    def load_config(self, path):
        if any(w and w.is_alive() for w in (self._camera, self._detector)):
            raise RuntimeError("stop camera before loading configuration")
        state = load_settings(path)
        if not state:
            return
        if state.get("schema_version") != 1:
            raise ValueError("unsupported settings schema_version")
        self.locator.restore_state(state.get("locator", {}))
        self._rgb_memory = copy.deepcopy(state.get("rgb_options", {}))
        self.gimbal.set_home(*state.get("home", [90, 90]))
        self.gimbal.set_tx_rate_hz(state.get("tx_rate_hz", 12))
        hz = float(state.get("detection_hz", 15))
        if not 1 <= hz <= 30:
            raise ValueError("detection_hz must be between 1 and 30")
        self.detection_hz = hz

    def close(self):
        try:
            self.stop_camera()
        finally:
            self.gimbal.disconnect()
        if self.config_path:
            self.save_config()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
