from __future__ import annotations
"""File-backed Legacy calibration editor. All widgets run on the GUI thread."""
import copy
import json
from pathlib import Path
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QMessageBox,
    QTabWidget, QGroupBox, QScrollArea)
from ..calibration import (CalibrationDocument, CalibrationChain, CalibrationError,
    calibration_readiness, matrix_rpy_deg, document_hash)


class DocumentEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, kind, settings):
        super().__init__()
        self.kind, self.settings, self.document = kind, settings, None
        self._loading = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_label = QLabel("未加载文件")
        self.path_label.setWordWrap(True)
        row.addWidget(self.path_label, 1)
        for label, callback in (("选择文件…", self.choose), ("重新读取", self.reload),
                                ("副本重置为文件值", self.reset)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.metadata = QLabel()
        self.metadata.setWordWrap(True)
        self.metadata.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.metadata)
        self.form_widget = QWidget()
        form = QFormLayout(self.form_widget)
        self.xyz, self.rpy = [], []
        for title, fields, lo, hi, decimals in (("translation / mm", self.xyz, -1e7, 1e7, 6),
                                               ("RPY / deg", self.rpy, -360, 360, 6)):
            row = QHBoxLayout()
            for axis in ("X", "Y", "Z"):
                row.addWidget(QLabel(axis))
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(decimals)
                spin.setSingleStep(0.1)
                spin.setKeyboardTracking(False)
                row.addWidget(spin)
                fields.append(spin)
                spin.valueChanged.connect(lambda _v, group=title: self.edit(group))
            form.addRow(title, row)
        self.model_widgets = {}
        if kind == "camera":
            row = QHBoxLayout()
            for key, values in (("pan_axis", ["z"]), ("tilt_axis", ["x", "y", "z"]),
                                ("pan_sign", [1.0, -1.0]), ("tilt_sign", [1.0, -1.0])):
                row.addWidget(QLabel(key))
                combo = QComboBox()
                for value in values:
                    combo.addItem(str(value), value)
                row.addWidget(combo)
                self.model_widgets[key] = combo
                combo.currentIndexChanged.connect(lambda _i: self.edit("model"))
            form.addRow("Legacy 轴和方向", row)
            row = QHBoxLayout()
            for key in ("pan_zero_deg", "tilt_zero_deg"):
                row.addWidget(QLabel(key))
                spin = QDoubleSpinBox()
                spin.setRange(-100000, 100000)
                spin.setDecimals(8)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda _v: self.edit("model"))
                row.addWidget(spin)
                self.model_widgets[key] = spin
            form.addRow("拟合零位 / Cmd unit≈deg", row)
        layout.addWidget(self.form_widget)
        self.matrix_label = QLabel()
        self.matrix_label.setWordWrap(True)
        self.matrix_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.matrix_label)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color:#b00020;")
        layout.addWidget(self.error)
        self.form_widget.setEnabled(False)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择标定 JSON", "", "JSON (*.json)")
        if path:
            self.load(path)

    def load(self, path, restore=False):
        try:
            doc = CalibrationDocument(path, self.kind)
            # Copies are restored only for exactly the same original contents.
            saved = self.settings.value(f"calibration_editor/{self.kind}_copy", "", type=str)
            if restore and saved:
                state = json.loads(saved)
                if state.get("fingerprint") == doc.fingerprint:
                    doc.parse(state["edited"])
                    doc.edited = state["edited"]
            self.document = doc
            self.error.clear()
            self.settings.setValue(f"calibration_editor/{self.kind}_path", doc.path)
            self.persist()
            self.render()
            self.changed.emit()
            return True
        except Exception as exc:
            # Fail closed: do not keep a previous chain underneath a failed load.
            self.document = None
            self.form_widget.setEnabled(False)
            self.path_label.setText(f"加载失败：{path}")
            self.metadata.clear()
            self.matrix_label.clear()
            self.error.setText(str(exc))
            self.changed.emit()
            return False

    def reload(self):
        if self.document:
            self.load(self.document.path)

    def reset(self):
        if self.document:
            self.document.reset()
            self.persist()
            self.render()
            self.changed.emit()

    def persist(self):
        if self.document:
            state = {"fingerprint": self.document.fingerprint, "edited": self.document.edited}
            self.settings.setValue(f"calibration_editor/{self.kind}_copy",
                                   json.dumps(state, ensure_ascii=False, allow_nan=False))
            self.settings.sync()

    def render(self):
        self._loading = True
        try:
            doc = self.document
            data = doc.edited
            parsed = doc.parse(data)
            self.path_label.setText(doc.path)
            self.form_widget.setEnabled(True)
            original_parsed = doc.parse(doc.original)
            original_t = original_parsed.t_tilt_camera if self.kind == "camera" else original_parsed.t_arm_panbase
            original_summary = (
                "文件原值 translation/mm=" + np.array2string(original_t[:3, 3]*1000, precision=5)
                + "；RPY/deg=" + np.array2string(matrix_rpy_deg(original_t[:3, :3]), precision=5)
                + ("；model=" + json.dumps(doc.original["model"], ensure_ascii=False) if self.kind == "camera" else ""))
            fit = doc.original.get("fit_summary", {})
            fit = {k: v for k, v in fit.items() if not isinstance(v, (list, dict))}
            self.metadata.setText(
                f"parent={parsed.parent_frame}  ←  child={parsed.child_frame}\n"
                + (f"source_pantilt_camera={doc.original.get('source_pantilt_camera', '未记录')}\n" if self.kind == "arm" else "")
                + "原文件 fit_summary（编辑后不重新评估）：" + json.dumps(fit, ensure_ascii=False) + "\n" + original_summary)
            transform = parsed.t_tilt_camera if self.kind == "camera" else parsed.t_arm_panbase
            for spin, val in zip(self.xyz, transform[:3, 3] * 1000):
                spin.setValue(float(val))
            for spin, val in zip(self.rpy, matrix_rpy_deg(transform[:3, :3])):
                spin.setValue(float(val))
            for key, widget in self.model_widgets.items():
                value = data["model"][key]
                if isinstance(widget, QComboBox):
                    widget.setCurrentIndex(widget.findData(value))
                else:
                    widget.setValue(float(value))
            self.render_matrix()
        finally:
            self._loading = False

    def render_matrix(self):
        parsed = self.document.parse(self.document.edited)
        transform = parsed.t_tilt_camera if self.kind == "camera" else parsed.t_arm_panbase
        rows = ["[" + ", ".join(f"{v:+.8f}" for v in row) + "]" for row in transform[:3, :3]]
        self.matrix_label.setText("副本旋转矩阵（随 RPY 编辑同步；Rz·Ry·Rx）：" + "  ".join(rows))

    def edit(self, group):
        if self._loading or self.document is None:
            return
        try:
            if group.startswith("translation"):
                self.document.update_pose(xyz_mm=[s.value() for s in self.xyz])
            elif group.startswith("RPY"):
                self.document.update_pose(rpy_deg=[s.value() for s in self.rpy])
            else:
                model = copy.deepcopy(self.document.edited["model"])
                for key, w in self.model_widgets.items():
                    model[key] = w.currentData() if isinstance(w, QComboBox) else w.value()
                self.document.update_pose(model=model)
            self.persist()
            self.render_matrix()
            self.error.clear()
            self.changed.emit()
        except Exception as exc:
            self.error.setText(f"编辑未生效：{exc}")


class CalibrationPanel(QGroupBox):
    chain_changed = pyqtSignal(object)

    def __init__(self, settings):
        super().__init__("坐标标定链 · Legacy 文件数据")
        self.settings = settings
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("运行数据源"))
        self.source = QComboBox()
        self.source.addItem("使用标定文件原始数据", "original")
        self.source.addItem("使用界面编辑后的数据", "edited")
        row.addWidget(self.source)
        self.show_editor = QPushButton("展开标定数据 / 编辑副本")
        self.show_editor.setCheckable(True)
        row.addWidget(self.show_editor)
        self.export_btn = QPushButton("将两份副本另存为 JSON…")
        row.addWidget(self.export_btn)
        row.addStretch()
        layout.addLayout(row)
        self.active_label = QLabel()
        self.active_label.setWordWrap(True)
        layout.addWidget(self.active_label)
        self.readiness_label = QLabel()
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)
        self.tabs = QTabWidget()
        self.editors = {kind: DocumentEditor(kind, settings) for kind in ("camera", "arm")}
        for kind, label in (("camera", "相机→云台"), ("arm", "云台→机械臂")):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.editors[kind])
            self.tabs.addTab(scroll, label)
            self.editors[kind].changed.connect(self.refresh_chain)
        self.tabs.setMaximumHeight(320)
        self.tabs.setMinimumHeight(270)
        layout.addWidget(self.tabs)
        self.tabs.hide()
        self.show_editor.toggled.connect(self.tabs.setVisible)
        note = QLabel("表单始终编辑工作副本；translation 显示 mm（JSON 为 m），RPY 显示 deg（JSON 为 rad）。原文件不会自动改写。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.source.currentIndexChanged.connect(self.refresh_chain)
        self.export_btn.clicked.connect(self.export_pair)

    def restore(self):
        # Remove retired geometry settings; Home is now independent of calibration.
        self.settings.remove("manual_calibration")
        self.settings.remove("calibration/arm_mode")
        source = self.settings.value("calibration_editor/source", "original", type=str)
        self.source.blockSignals(True)
        self.source.setCurrentIndex(max(0, self.source.findData(source)))
        self.source.blockSignals(False)
        for kind, old_key in (("camera", "camera_to_gimbal"), ("arm", "gimbal_to_arm")):
            path = self.settings.value(f"calibration_editor/{kind}_path", "", type=str)
            if not path:
                legacy = self.settings.value(f"calibration/{old_key}", "", type=str)
                if legacy:
                    path = legacy
            if not path:
                default_name = ("camera_to_gimbal_default.json" if kind == "camera"
                                else "cr10a_arm_to_pantilt_charuco_calibration.json")
                path = str(Path(__file__).resolve().parents[1] / "assets" / "calibration" / default_name)
            if path:
                self.editors[kind].load(path, restore=True)
        self.refresh_chain()

    def refresh_chain(self, *_):
        source = self.source.currentData()
        self.settings.setValue("calibration_editor/source", source)
        self.settings.sync()
        cam, arm = (self.editors[k].document for k in ("camera", "arm"))
        message, blocked = calibration_readiness(cam, arm)
        chain = CalibrationChain(active_source=source)
        try:
            if cam:
                chain.camera = cam.parse(cam.active(source))
            if arm:
                chain.arm = arm.parse(arm.active(source))
        except Exception as exc:
            message, blocked = str(exc), True
        chain.blocked_reason = message if blocked else ""
        changed = any(d and document_hash(d.original) != document_hash(d.edited) for d in (cam, arm))
        self.active_label.setText("当前生效：" + self.source.currentText()
            + (" · 副本已有修改" if changed else " · 副本与文件值一致")
            + " · Legacy 使用 Pan/Tilt Cmd；编码器仅监视")
        self.readiness_label.setText(message)
        self.readiness_label.setStyleSheet("color:#b00020;font-weight:600;" if blocked else "color:#187a2f;")
        self.export_btn.setEnabled(cam is not None and arm is not None and not blocked)
        self.chain_changed.emit(chain)

    def bind_locator(self, locator):
        """One-way UI -> core binding. Call once after constructing this panel."""
        def sync(_chain):
            locator.set_calibration_documents(self.editors["camera"].document,
                self.editors["arm"].document, self.source.currentData())
        self.chain_changed.connect(sync)
        sync(None)

    def export_pair(self):
        cam, arm = (self.editors[k].document for k in ("camera", "arm"))
        if not cam or not arm or calibration_readiness(cam, arm)[1]:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择保存位置（自动新建子文件夹）")
        if not directory:
            return
        try:
            import datetime
            import uuid
            dest = Path(directory) / ("calibration_edited_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
            dest.mkdir()
            camera_data, arm_data = copy.deepcopy(cam.edited), copy.deepcopy(arm.edited)
            camera_path = dest / "pantilt_camera_calibration.json"
            arm_data["source_pantilt_camera"] = camera_path.name
            arm_data["source_pantilt_camera_sha256"] = document_hash(camera_data)
            for p, data in ((camera_path, camera_data), (dest / "arm_to_pantilt_calibration.json", arm_data)):
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            QMessageBox.information(self, "已另存副本", f"已保存到：{dest}\n当前加载文件与运行数据源保持原选择。fit_summary 是原标定记录，不代表修改后的精度。")
        except Exception as exc:
            QMessageBox.warning(self, "另存失败", str(exc))
