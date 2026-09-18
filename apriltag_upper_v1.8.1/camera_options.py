"""RGB hardware controls isolated from capture and detection."""
import math
import threading
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QDoubleSpinBox, QComboBox, QPushButton, QWidget)

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


class RGBOptionsWorker(QThread):
    snapshot_ready = Signal(object)

    def __init__(self, sensor, option_enum, serial):
        super().__init__()
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


class RGBOptionsPanel(QGroupBox):
    requested = Signal(object)

    def __init__(self, settings):
        super().__init__("D435 RGB 硬件参数")
        self.settings = settings
        self.serial = ""
        self._restored = False
        self.records, self.widgets = {}, {}
        self._pending = set()
        self.minimum_generation = 0
        self.last_errors = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.toggle = QPushButton("展开相机参数")
        self.toggle.setCheckable(True)
        row.addWidget(self.toggle)
        self.status = QLabel("启动相机后读取本机实际支持项和范围")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        layout.addLayout(row)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        layout.addWidget(self.body)
        self.body.hide()
        self.toggle.toggled.connect(self.body.setVisible)
        self.note = QLabel("成功写入的参数按相机序列号保存并于下次启动恢复。自动模式开启时，相应手动参数不可调。自动曝光优先级可能降低帧率。")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

    def reset(self):
        self.serial = ""
        self._restored = False
        self._pending.clear()
        self.minimum_generation = 0
        self.last_errors = []
        self.body.setEnabled(False)
        self.status.setText("相机未运行；启动后读取硬件参数")

    def submit(self, key, value):
        self._pending.add(key)
        self.widgets[key].setEnabled(False)
        self.requested.emit({key: value})

    def update_snapshot(self, payload):
        # Every acknowledged write must be persisted, including writes whose
        # visual snapshot is superseded by a newer GUI request.
        serial = payload["serial"]
        for key, value in payload["applied"].items():
            self.settings.setValue(f"rgb_options/{serial}/{key}", value)
        if payload["applied"]:
            self.settings.sync()
        if payload["generation"] < self.minimum_generation:
            return
        records = payload["records"]
        self.body.setEnabled(True)
        rebuild = serial != self.serial or set(records) != set(self.records)
        self.serial, self.records = serial, records
        self._pending.clear()
        if payload["errors"]:
            self.last_errors = payload["errors"]
        elif payload["applied"]:
            self.last_errors = []
        if rebuild:
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.widgets = {}
            for i, (key, entry) in enumerate(records.items()):
                row, col = i // 3, (i % 3) * 2
                label = QLabel(OPTION_LABELS[key])
                self.grid.addWidget(label, row, col)
                if key in AUTO_KEYS or key == "auto_exposure_priority" or entry.get("choices"):
                    widget = QComboBox()
                    choices = entry.get("choices", [(0.0, "关闭"), (1.0, "开启")])
                    for v, text in choices:
                        widget.addItem(text, v)
                    widget.currentIndexChanged.connect(lambda _i, k=key, w=widget: self.submit(k, w.currentData()))
                else:
                    widget = QDoubleSpinBox()
                    widget.setDecimals(6)
                    widget.setKeyboardTracking(False)
                    widget.valueChanged.connect(lambda v, k=key: self.submit(k, v))
                self.widgets[key] = widget
                self.grid.addWidget(widget, row, col + 1)
        for key, widget in self.widgets.items():
            e = records[key]
            if "error" in e:
                widget.setEnabled(False)
                widget.setToolTip(e["error"])
                continue
            dependency = DEPENDENCIES.get(key)
            auto = records.get(dependency, {}).get("value", 0) != 0
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findData(e["value"]))
            else:
                widget.setRange(e["min"], e["max"])
                widget.setSingleStep(e["step"] or 0.001)
                # Don't replace text while the user is typing.
                if not widget.hasFocus() or key in payload["applied"] or payload["errors"]:
                    widget.setValue(e["value"])
            widget.blockSignals(False)
            widget.setEnabled(not e["readonly"] and not auto)
            widget.setToolTip(f"SDK范围 {e['min']:g} … {e['max']:g}；步进 {e['step']:g}；实际值 {e['value']:g}\n{e['description']}"
                              + ("\n请先关闭自动模式" if auto else ""))
        errors = self.last_errors + [f"{OPTION_LABELS[k]}：{e['error']}" for k, e in records.items() if "error" in e]
        self.status.setText(("；".join(errors) if errors else f"S/N {serial} · 已读取 {len(records)} 个支持项")
                            if records else "当前 RGB sensor 未报告可用选项")
        self.status.setStyleSheet("color:#b00020;" if errors else "")
        if not self._restored:
            self._restored = True
            pending = {}
            for key, entry in records.items():
                saved = self.settings.value(f"rgb_options/{serial}/{key}")
                if saved is not None and "error" not in entry and not entry["readonly"]:
                    try:
                        pending[key] = float(saved)
                    except (TypeError, ValueError):
                        pass
            for key, dependency in DEPENDENCIES.items():
                if pending.get(dependency, records.get(dependency, {}).get("value", 0)):
                    pending.pop(key, None)
            if pending:
                self.requested.emit(pending)
