from __future__ import annotations
import math
import threading
from .runtime import ThreadWorker, EventHook

OPTION_LABELS = {
    "enable_auto_exposure": "自动曝光", "exposure": "曝光 / SDK单位",
    "gain": "增益", "enable_auto_white_balance": "自动白平衡",
    "white_balance": "白平衡 / K", "brightness": "亮度", "contrast": "对比度",
    "saturation": "饱和度", "sharpness": "锐度", "gamma": "Gamma",
    "hue": "色调", "backlight_compensation": "背光补偿",
    "power_line_frequency": "电源频率", "auto_exposure_priority": "自动曝光优先级",
}
DEPENDENCIES = {"exposure": "enable_auto_exposure", "gain": "enable_auto_exposure",
                "white_balance": "enable_auto_white_balance"}
AUTO_KEYS = ("enable_auto_exposure", "enable_auto_white_balance")


class SensorOptions:
    """All SDK calls are made by one option thread; no GUI or capture calls."""
    def __init__(self, sensor, option_enum):
        self.sensor = sensor
        self.options = {}
        for name in OPTION_LABELS:
            option = getattr(option_enum, name, None)
            try:
                if option is not None and sensor.supports(option):
                    self.options[name] = option
            except Exception:
                pass

    def snapshot(self):
        result = {}
        for name, opt in self.options.items():
            try:
                r = self.sensor.get_option_range(opt)
                value = float(self.sensor.get_option(opt))
                entry = dict(min=float(r.min), max=float(r.max), step=float(r.step),
                             default=float(r.default), value=value,
                             readonly=bool(self.sensor.is_option_read_only(opt)))
                if not all(math.isfinite(entry[k]) for k in ("min", "max", "step", "default", "value")):
                    continue
                if entry["max"] < entry["min"]:
                    continue
                try:
                    entry["description"] = self.sensor.get_option_description(opt)
                except Exception:
                    entry["description"] = ""
                if name == "power_line_frequency":
                    choices = []
                    step = entry["step"] or 1
                    for i in range(min(16, int((entry["max"] - entry["min"]) / step) + 1)):
                        v = entry["min"] + i * step
                        try:
                            label = self.sensor.get_option_value_description(opt, v)
                        except Exception:
                            label = None
                        choices.append((v, label or str(v)))
                    entry["choices"] = choices
                result[name] = entry
            except Exception as exc:
                result[name] = {"error": str(exc)}
        return result

    def apply(self, pending):
        applied, errors = {}, []
        ordered = [k for k in AUTO_KEYS if k in pending] + [k for k in pending if k not in AUTO_KEYS]
        for key in ordered:
            try:
                if key not in self.options:
                    raise ValueError("当前 RGB sensor 不支持此选项")
                opt = self.options[key]
                if self.sensor.is_option_read_only(opt):
                    raise ValueError("该参数当前只读")
                dependency = DEPENDENCIES.get(key)
                if dependency in self.options and self.sensor.get_option(self.options[dependency]) != 0:
                    # Keep a remembered manual value for a later manual session;
                    # never silently disable auto mode to restore it.
                    raise ValueError("请先关闭相应自动模式")
                value = float(pending[key])
                if not math.isfinite(value):
                    raise ValueError("参数必须是有限数值")
                r = self.sensor.get_option_range(opt)
                if value < r.min or value > r.max:
                    raise ValueError(f"超出当前范围 [{r.min}, {r.max}]")
                if r.step > 0:
                    value = r.min + round((value - r.min) / r.step) * r.step
                    value = min(r.max, max(r.min, value))
                self.sensor.set_option(opt, value)
                applied[key] = float(self.sensor.get_option(opt))
            except Exception as exc:
                errors.append(f"{OPTION_LABELS.get(key, key)}：{exc}")
        return applied, errors


class RGBOptionsWorker(ThreadWorker):

    def __init__(self, sensor, option_enum, serial):
        super().__init__()
        self.snapshot_ready = EventHook()
        self.controller = SensorOptions(sensor, option_enum)
        self.serial = serial
        self._stop_event, self._wake = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._pending = {}
        self._generation = 0

    def request(self, values):
        with self._lock:
            self._pending.update(values)
            self._generation += 1
            generation = self._generation
        self._wake.set()
        return generation

    def stop(self):
        self._stop_event.set()
        self._wake.set()

    def run(self):
        while not self._stop_event.is_set():
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, {}
                generation = self._generation
            applied, errors = self.controller.apply(pending)
            records = self.controller.snapshot()
            if not self._stop_event.is_set():
                self.snapshot_ready.emit(dict(serial=self.serial, records=records,
                    applied=applied, errors=errors, generation=generation))
            self._wake.wait(1.0)


