from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QLabel, QDoubleSpinBox, QComboBox, QPushButton, QWidget)
from ..camera_options import OPTION_LABELS, AUTO_KEYS, DEPENDENCIES

class RGBOptionsPanel(QGroupBox):
    requested = pyqtSignal(object)

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
