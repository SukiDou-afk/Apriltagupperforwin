from __future__ import annotations
import time
import threading
import traceback
import numpy as np
import cv2
import pyrealsense2 as rs
from .runtime import ThreadWorker, EventHook
from .camera_options import RGBOptionsWorker
DEFAULT_WIDTH=1280
DEFAULT_HEIGHT=720
DEFAULT_FPS=30

def _rs_format(name):
    """Return a RealSense format enum when this SDK version exposes it."""
    try:
        return getattr(rs.format, name)
    except Exception:
        return None


def profile_format_name(fmt):
    names = (
        ("bgr8", "BGR8"),
        ("rgb8", "RGB8"),
        ("yuyv", "YUYV"),
        ("uyvy", "UYVY"),
        ("mjpeg", "MJPEG"),
        ("bgra8", "BGRA8"),
        ("rgba8", "RGBA8"),
    )
    for attr, label in names:
        value = _rs_format(attr)
        if value is not None and fmt == value:
            return label
    return str(fmt)


def profile_format_key(fmt):
    """Stable string key used to persist a RealSense stream profile."""
    names = ("bgr8", "rgb8", "yuyv", "uyvy", "mjpeg", "bgra8", "rgba8")
    for attr in names:
        value = _rs_format(attr)
        if value is not None and fmt == value:
            return attr
    return str(fmt)


def profile_key(data):
    if not data:
        return ""
    w, h, fps, fmt = data
    return f"{int(w)}x{int(h)}@{int(fps)}:{profile_format_key(fmt)}"


def _supported_color_formats():
    """Formats that this program can convert to OpenCV BGR frames."""
    names = ("bgr8", "rgb8", "yuyv", "uyvy", "mjpeg", "bgra8", "rgba8")
    out = []
    for name in names:
        value = _rs_format(name)
        if value is not None:
            out.append(value)
    return tuple(out)


def color_frame_to_bgr(color_frame, fallback_format=None):
    """Convert common RealSense color formats to an OpenCV BGR image.

    RealSense Viewer can expose 1920x1080@30 through native YUYV/MJPEG on
    some D435 firmware/driver combinations.  Restricting the application to
    RGB8/BGR8 therefore hid valid high-resolution profiles.  This conversion
    keeps the native profile and converts only after capture.
    """
    try:
        fmt = color_frame.profile.format()
    except Exception:
        fmt = fallback_format

    data = np.asanyarray(color_frame.get_data())
    bgr8 = _rs_format("bgr8")
    rgb8 = _rs_format("rgb8")
    yuyv = _rs_format("yuyv")
    uyvy = _rs_format("uyvy")
    mjpeg = _rs_format("mjpeg")
    bgra8 = _rs_format("bgra8")
    rgba8 = _rs_format("rgba8")

    if bgr8 is not None and fmt == bgr8:
        if data.ndim == 3 and data.shape[2] == 3:
            return data
    if rgb8 is not None and fmt == rgb8:
        if data.ndim == 3 and data.shape[2] == 3:
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if yuyv is not None and fmt == yuyv:
        if data.ndim == 3 and data.shape[2] == 2:
            return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUY2)
    if uyvy is not None and fmt == uyvy:
        if data.ndim == 3 and data.shape[2] == 2:
            return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_UYVY)
    if bgra8 is not None and fmt == bgra8:
        if data.ndim == 3 and data.shape[2] == 4:
            return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    if rgba8 is not None and fmt == rgba8:
        if data.ndim == 3 and data.shape[2] == 4:
            return cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
    if mjpeg is not None and fmt == mjpeg:
        encoded = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded

    # Some SDK builds expose a converted stream whose enum is unexpected.
    # Accept an already-decoded 3-channel frame as a safe last resort.
    if data.ndim == 3 and data.shape[2] == 3:
        return data
    raise ValueError(f"Unsupported RealSense color frame format: {profile_format_name(fmt)} shape={data.shape}")


def discover_color_profiles(serial=None):
    """Return (device_name, serial, [(w, h, fps, format), ...]).

    Include native YUYV/MJPEG as well as RGB/BGR.  This is important on D435
    systems where RealSense Viewer offers 1920x1080@30 but the sensor does not
    advertise that mode as RGB8/BGR8 directly.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        return None, None, []

    chosen = None if serial else devices[0]
    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name)
            if ((serial and dev.get_info(rs.camera_info.serial_number) == serial) or
                    (not serial and ("D435" in name.upper() or "D4" in name.upper()))):
                chosen = dev
                break
        except Exception:
            pass

    if chosen is None:
        return None, serial, []
    try:
        name = chosen.get_info(rs.camera_info.name)
    except Exception:
        name = "RealSense"
    try:
        serial = chosen.get_info(rs.camera_info.serial_number)
    except Exception:
        serial = "Unknown"

    allowed = set(_supported_color_formats())
    rank_names = ["bgr8", "rgb8", "yuyv", "uyvy", "mjpeg", "bgra8", "rgba8"]
    preferred_rank = {}
    for idx, fmt_name in enumerate(rank_names):
        fmt = _rs_format(fmt_name)
        if fmt is not None:
            preferred_rank[fmt] = idx

    found = {}
    for sensor in chosen.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                if profile.stream_type() != rs.stream.color:
                    continue
                fmt = profile.format()
                if fmt not in allowed:
                    continue
                vp = profile.as_video_stream_profile()
                key = (int(vp.width()), int(vp.height()), int(profile.fps()))
                old = found.get(key)
                if old is None or preferred_rank.get(fmt, 99) < preferred_rank.get(old, 99):
                    found[key] = fmt
            except Exception:
                continue

    profiles = [(w, h, fps, fmt) for (w, h, fps), fmt in found.items()]
    profiles.sort(key=lambda x: (
        -(x[0] * x[1]),
        -x[2],
        -x[0],
        -x[1],
        preferred_rank.get(x[3], 99),
    ))
    return name, serial, profiles


class CameraCaptureWorker(ThreadWorker):
    """RealSense capture only: no AprilTag, no drawing, no Qt image conversion."""


    def __init__(
        self,
        frame_buffer,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
        pixel_format=rs.format.bgr8,
        parent=None,
        serial=None,
        pose_provider=None,
    ):
        super().__init__(parent)
        self.status = EventHook()
        self.error = EventHook()
        self.camera_info = EventHook()
        self.stats = EventHook()
        self.rgb_options = EventHook()
        self._running = False
        self._buffer = frame_buffer
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pixel_format = pixel_format
        self.serial = serial
        self.pose_provider = pose_provider
        self._options_worker = None
        self._stop_requested = threading.Event()

    def request_options(self, values):
        worker = self._options_worker
        return worker.request(values) if worker is not None else None

    def stop(self):
        self._stop_requested.set()
        self._running = False
        if self._options_worker:
            self._options_worker.stop()

    def run(self):
        if self._stop_requested.is_set():
            return
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self._width, self._height,
                                 self._pixel_format, self._fps)
            profile = pipeline.start(config)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(
                f"D435 彩色相机启动失败：{e}\n\n"
                f"当前选择：{self._width}×{self._height}@{self._fps} "
                f"{profile_format_name(self._pixel_format)}\n"
                "请点击‘刷新分辨率’，选择相机实际支持的项目后重试。"
            )
            return

        self._running = True
        fps_t0 = time.monotonic()
        fps_frames = 0
        try:
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "RealSense"
            serial = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else "未知"

            try:
                rgb_sensor = next(sensor for sensor in device.query_sensors()
                                  if any(p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles()))
                self._options_worker = RGBOptionsWorker(rgb_sensor, rs.option, serial)
                self._options_worker.snapshot_ready.connect(self.rgb_options.emit)
                self._options_worker.start()
            except Exception as exc:
                self.status.emit(f"RGB 参数面板初始化失败（采集继续）：{exc}")

            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            camera_matrix = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1, 1)

            actual_format = color_profile.format()
            actual_w = int(color_profile.width())
            actual_h = int(color_profile.height())
            actual_fps = int(color_profile.fps())
            self._buffer.set_stream_info(
                camera_matrix,
                dist_coeffs,
                width=actual_w,
                height=actual_h,
                fps=actual_fps,
                format_name=profile_format_name(actual_format),
            )

            self.camera_info.emit(
                f"{name}  S/N {serial}  |  {actual_w}×{actual_h}@{actual_fps} "
                f"{profile_format_name(actual_format)}  |  "
                f"fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}"
            )
            self.status.emit("相机采集线程已启动；预览与 AprilTag 识别已与采集解耦。")

            consecutive_timeouts = 0
            while self._running and not self._stop_requested.is_set():
                try:
                    frames = pipeline.wait_for_frames(1000)
                except Exception as exc:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 5 and not self._stop_requested.is_set():
                        raise RuntimeError("连续 5 次未收到相机帧：" + str(exc)) from exc
                    continue
                consecutive_timeouts = 0
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                try:
                    frame = color_frame_to_bgr(color_frame, actual_format)
                except Exception as exc:
                    self.status.emit(f"彩色帧转换失败：{exc}")
                    continue

                # RealSense native BGR can be a view over SDK-owned memory.  Keep
                # one owned latest frame so it remains valid after color_frame is
                # released.  Converted YUYV/RGB/MJPEG frames are usually already
                # owned; avoid another copy in that common case.
                if frame.base is not None or not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame).copy()

                now = time.monotonic()
                seq = self._buffer.publish(frame, now, **(self.pose_provider() if self.pose_provider else {}))
                fps_frames += 1
                elapsed = now - fps_t0
                if elapsed >= 1.0:
                    self.stats.emit({
                        "input_fps": fps_frames / elapsed,
                        "seq": seq,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    })
                    fps_t0 = now
                    fps_frames = 0

        except Exception as e:
            traceback.print_exc()
            self.error.emit(
                f"相机采集线程异常：{e}\n\n"
                "详细 Python traceback 已输出到启动本程序的命令行窗口。"
            )
        finally:
            if self._options_worker:
                self._options_worker.stop()
                # Keep the sensor alive until the option thread has released it.
                self._options_worker.wait()
                self._options_worker = None
            try:
                pipeline.stop()
            except Exception:
                pass
            self._running = False
            self.status.emit("相机采集线程已停止")
