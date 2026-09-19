from __future__ import annotations
import sys
import time
import traceback
import json
import threading
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QSettings, QTimer, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from ..pantilt import PantiltError, PantiltSerial
from ..calibration import CalibrationChain
from .calibration_editor import CalibrationPanel
from .camera_options import RGBOptionsPanel

from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget, QHeaderView, QSpinBox, QGridLayout, QFileDialog, QLineEdit, QFormLayout,
    QScrollArea, QSizePolicy
)

from ..frames import LatestFrameBuffer
from ..camera import (_rs_format, profile_format_name, profile_format_key,
                      profile_key, discover_color_profiles, color_frame_to_bgr)
from ..detection import build_object_points
from ..drawing import draw_overlays_inplace
from .workers import CameraCaptureWorker, DetectionWorker

APP_TITLE = "AprilTag 目标点定位 + 云台/机械臂坐标换算 v2.0.0 PyQt6"
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


class FullscreenPreviewWindow(QWidget):
    """Optional camera-only fullscreen window. ESC closes it."""

    closed = pyqtSignal()

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("AprilTag 相机全屏预览")
        self.setStyleSheet("background:#000;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("等待相机画面\nEsc 退出全屏")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background:#000; color:#ddd; font-size:20px;")
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.label)

    def set_pixmap(self, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        size = self.label.contentsRect().size()
        if size.width() > 1 and size.height() > 1:
            pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


from .views import FrameView3D


class MainWindow(QMainWindow):
    tag_size_changed = pyqtSignal(float)
    median_changed = pyqtSignal(bool)
    target_offsets_changed = pyqtSignal(object)
    results_updated = pyqtSignal(object)

    def __init__(self, settings_path=None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 900)
        # Portable INI stored in the launch working directory. All user-adjustable
        # detection/camera options are restored automatically on next launch.
        settings_path = str(settings_path or (Path.cwd() / "settings.ini"))
        self.settings = QSettings(settings_path, QSettings.Format.IniFormat)
        self.frame_buffer = LatestFrameBuffer()
        self.camera_worker = None
        self._camera_stopping = False
        self._close_pending = False
        self.detector_worker = None
        self.pantilt = PantiltSerial()
        self.calibration = CalibrationChain()
        self.last_frame = None
        self.last_results = []
        self._latest_detection_captured_at = 0.0
        self.latest_overlays = []
        self._latest_detection_received = 0.0
        self._latest_detection_frame_seq = -1
        self._capture_fps_value = 0.0
        self._preview_fps_t0 = time.monotonic()
        self._preview_fps_frames = 0
        self._preview_fps_value = 0.0
        self._detection_fps_value = 0.0
        self._detection_ms_value = 0.0
        self._last_preview_seq = -1
        self._last_preview_size = (0, 0)
        self._last_result_ui_update = 0.0
        self.fullscreen_preview = None
        self.primary_result_id = None
        self._latest_display_records = {}
        self.detected_tag_ids = set()
        self.tag_offsets_mm = {}
        self.feedback_pan = None
        self.feedback_tilt = None
        self.feedback_valid = False
        self._feedback_online_prev = False
        # v1.7 continuous pan/tilt control state. Button presses update only the
        # newest target; serial transmission is independently rate-limited.
        self._pantilt_hold_dir = (0, 0)
        self._pantilt_hold_started = False
        self._pantilt_hold_float_pan = 90.0
        self._pantilt_hold_float_tilt = 90.0
        self._pantilt_hold_last_t = 0.0
        self._pantilt_last_tx_error = ""
        self._loading_profiles = False
        self._build_ui()
        self._restore_saved_options()
        self._load_saved_calibrations()
        self.rx_timer = QTimer(self)
        self.rx_timer.setInterval(50)
        self.rx_timer.timeout.connect(self.poll_pantilt_rx)
        self.rx_timer.start()
        self.pantilt_hold_delay_timer = QTimer(self)
        self.pantilt_hold_delay_timer.setSingleShot(True)
        self.pantilt_hold_delay_timer.setInterval(260)
        self.pantilt_hold_delay_timer.timeout.connect(self._begin_pantilt_hold_repeat)
        self.pantilt_hold_repeat_timer = QTimer(self)
        self.pantilt_hold_repeat_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.pantilt_hold_repeat_timer.setInterval(50)
        self.pantilt_hold_repeat_timer.timeout.connect(self._pantilt_hold_tick)
        self.preview_timer = QTimer(self)
        self.preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.preview_timer.setInterval(16)
        self.preview_timer.timeout.connect(self.refresh_preview)
        saved_geometry = self.settings.value("window/geometry")
        if saved_geometry is not None:
            try:
                self.restoreGeometry(saved_geometry)
            except Exception:
                pass
        self._append_log("程序已启动。请确认使用的是 AprilTag 36h11 标签。")
        self._append_log("XYZ 坐标由 RealSense 彩色相机内参 + AprilTag PnP 计算，不使用深度流。")
        self._append_log("v2.0.0：PyQt6 界面；相机、识别、标定及串口功能共用 apriltag_sdk。")
        self._append_log("v1.5：相机采集、AprilTag识别、GUI预览已拆成独立执行链；任何识别耗时都不再阻塞 D435 采集。")
        self._append_log("v1.5：预览只读取最新帧，GUI来不及显示的旧帧直接丢弃，不再形成越来越大的显示延迟。")
        self._append_log("v1.5：默认适应窗口时先缩小BGR图再转QImage；可切换1:1原始像素或全屏预览。")
        self._append_log("v1.6：新增云台编码器GBK主动反馈后台读取；舵机A=Pan，舵机B=Tilt，并显示 Cmd / Encoder / Enc-Cmd 误差。")
        self._append_log("v1.6：Legacy 坐标换算严格继续使用 Pan/Tilt Cmd，不让编码器反馈改变旧标定模型；反馈目前仅用于监视和误差诊断。")
        self._append_log("v1.7：云台改为长按连续运动；单击仍按步长点动，按住约260 ms后按连续速度持续改变目标。")
        self._append_log("v1.7：串口发送改为独立限频线程，默认最多12 Hz；只发送最新目标，旧中间目标不排队；每个目标只发1帧。")
        self._append_log("目标点偏移按 AprilTag ID 独立配置：每个 ID 都可有自己的 XYZ；未配置 ID 默认目标点=Tag中心。")
        self._append_log("偏移坐标定义在各自 Tag 坐标系：+X 向 Tag 右，+Y 向 Tag 上，+Z 垂直 Tag 正面向外。")
        self._append_log("v1.8.1：标定来源字段仅供查看，不按文件名、路径或来源指纹限制组合；Legacy 始终使用 Cmd。")
        self._append_log("坐标方向：Camera +X右/+Y下/+Z前；Base XYZ遵循机械臂基坐标。")
        self._append_log("编码器协议已确认：串口主动回传 GBK 文本‘舵机A角度：xxx / 舵机B角度：yyy’，其中 A=Pan、B=Tilt。")
        self.refresh_profiles(show_dialog=False)

    def _build_ui(self):
        # Put the whole work area in a two-direction scroll area so the program
        # remains usable on smaller monitors or Windows display scaling >100%.
        self.page_scroll = QScrollArea()
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        central = QWidget()
        central.setMinimumSize(1120, 820)
        self.page_scroll.setWidget(central)
        self.setCentralWidget(self.page_scroll)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        controls = QGroupBox("检测设置")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(8, 5, 8, 5)
        controls_layout.setSpacing(6)

        controls_layout.addWidget(QLabel("Tag 家族："))
        family_label = QLabel("AprilTag 36h11")
        family_label.setStyleSheet("font-weight: 600;")
        controls_layout.addWidget(family_label)

        controls_layout.addSpacing(14)
        controls_layout.addWidget(QLabel("实际边长："))
        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(5.0, 1000.0)
        self.tag_size_spin.setDecimals(1)
        self.tag_size_spin.setSingleStep(5.0)
        self.tag_size_spin.setValue(100.0)
        self.tag_size_spin.setSuffix(" mm")
        self.tag_size_spin.setToolTip("填写打印后 AprilTag 有效方形区域的实际边长")
        controls_layout.addWidget(self.tag_size_spin)

        controls_layout.addSpacing(14)
        controls_layout.addWidget(QLabel("分辨率："))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(185)
        self.profile_combo.setToolTip("停止相机后可以切换；列表来自当前 RealSense 彩色相机")
        controls_layout.addWidget(self.profile_combo)

        self.refresh_btn = QPushButton("刷新分辨率")
        self.refresh_btn.setToolTip("重新读取当前 RealSense 支持的彩色分辨率/帧率")
        controls_layout.addWidget(self.refresh_btn)

        self.median_box = QCheckBox("5帧中值稳定")
        self.median_box.setChecked(True)
        controls_layout.addWidget(self.median_box)

        controls_layout.addWidget(QLabel("识别Hz："))
        self.detect_hz_spin = QDoubleSpinBox()
        self.detect_hz_spin.setRange(1.0, 30.0)
        self.detect_hz_spin.setDecimals(0)
        self.detect_hz_spin.setSingleStep(1.0)
        self.detect_hz_spin.setValue(15.0)
        self.detect_hz_spin.setToolTip("AprilTag识别线程的目标频率；不会改变D435采集帧率")
        self.detect_hz_spin.setMaximumWidth(72)
        controls_layout.addWidget(self.detect_hz_spin)

        controls_layout.addWidget(QLabel("预览："))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItem("适应窗口", "fit")
        self.preview_mode_combo.addItem("1:1原始像素", "1to1")
        self.preview_mode_combo.setToolTip("适应窗口优先性能；1:1保持原始像素并使用局部滚动条")
        controls_layout.addWidget(self.preview_mode_combo)

        controls_layout.addStretch(1)
        self.fullscreen_btn = QPushButton("全屏预览")
        self.start_btn = QPushButton("启动相机")
        self.stop_btn = QPushButton("停止相机")
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.fullscreen_btn)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        root.addWidget(controls)

        offset_box = QGroupBox("ID → 目标点偏移（每个 AprilTag 独立配置）")
        offset_root = QVBoxLayout(offset_box)
        offset_root.setContentsMargins(8, 5, 8, 5)
        offset_root.setSpacing(4)

        offset_layout = QHBoxLayout()
        offset_layout.setSpacing(6)
        offset_layout.addWidget(QLabel("ID："))
        self.offset_id_combo = QComboBox()
        self.offset_id_combo.setEditable(True)
        self.offset_id_combo.setMinimumWidth(90)
        self.offset_id_combo.setToolTip("可直接输入 tag36h11 的 ID，也可从已配置/已检测 ID 中选择")
        offset_layout.addWidget(self.offset_id_combo)
        self.offset_select_btn = QPushButton("选择/新增")
        self.offset_select_btn.setToolTip("选择该 ID；若尚未配置，则以 0/0/0 mm 新建并立即保存")
        offset_layout.addWidget(self.offset_select_btn)
        offset_layout.addWidget(QLabel("X："))
        self.offset_x_spin = self._make_offset_spin()
        offset_layout.addWidget(self.offset_x_spin)
        offset_layout.addWidget(QLabel("Y："))
        self.offset_y_spin = self._make_offset_spin()
        offset_layout.addWidget(self.offset_y_spin)
        offset_layout.addWidget(QLabel("Z："))
        self.offset_z_spin = self._make_offset_spin()
        offset_layout.addWidget(self.offset_z_spin)
        self.offset_delete_btn = QPushButton("删除此ID")
        self.offset_delete_btn.setToolTip("删除该 ID 的专属偏移；删除后识别此 ID 时目标点回到 Tag 中心")
        offset_layout.addWidget(self.offset_delete_btn)
        offset_layout.addStretch(1)
        offset_root.addLayout(offset_layout)

        offset_bottom = QHBoxLayout()
        offset_bottom.setSpacing(8)
        self.offset_table = QTableWidget(0, 4)
        self.offset_table.setHorizontalHeaderLabels(["ID", "X/mm", "Y/mm", "Z/mm"])
        self.offset_table.verticalHeader().setVisible(False)
        self.offset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.offset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.offset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.offset_table.setMaximumHeight(105)
        self.offset_table.setMinimumWidth(390)
        self.offset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        offset_bottom.addWidget(self.offset_table, 1)
        offset_note = QLabel(
            "每个 ID 独立保存 XYZ 偏移。未配置 ID 默认 [0,0,0]，即目标点=Tag 中心。\n"
            "+X：Tag 图案向右；+Y：Tag 图案向上；+Z：垂直 Tag 正面向外。检测到的 ID 会自动加入 ID 下拉框。"
        )
        offset_note.setWordWrap(True)
        offset_note.setStyleSheet("color:#555;")
        offset_bottom.addWidget(offset_note, 2)
        offset_root.addLayout(offset_bottom)
        root.addWidget(offset_box)

        self.rgb_options_panel = RGBOptionsPanel(self.settings)
        root.addWidget(self.rgb_options_panel)
        self.rgb_options_panel.requested.connect(self.request_rgb_options)
        self.calibration_panel = CalibrationPanel(self.settings)
        self.calibration_panel.chain_changed.connect(self.on_calibration_chain_changed)
        root.addWidget(self.calibration_panel)
        self.coord_source_label = QLabel("Legacy 使用 Cmd；编码器仅监视")
        self.coord_source_label.setWordWrap(True)
        root.addWidget(self.coord_source_label)
        self.active_source_status = QLabel()
        self.statusBar().addPermanentWidget(self.active_source_status)

        main_area = QHBoxLayout()
        main_area.setSpacing(6)
        root.addLayout(main_area, 1)

        self.video_label = QLabel("点击“启动相机”")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        # Critical: ignore pixmap size hints.  Otherwise QLabel's pixmap can feed
        # back into the layout and make the preview panel grow a few pixels on
        # every update inside the scroll area.
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setScaledContents(False)
        self.video_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.video_label.setStyleSheet("background:#151515; color:#dddddd; font-size:18px;")
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidget(self.video_label)
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setMinimumSize(640, 360)
        self.video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.video_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.video_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self.video_scroll, 4)

        spatial_box = QGroupBox("空间坐标关系（机械臂 Base 参考）")
        spatial_layout = QVBoxLayout(spatial_box)
        spatial_layout.setContentsMargins(6, 5, 6, 5)
        self.spatial_view = FrameView3D()
        self.spatial_view.setMinimumHeight(250)
        spatial_layout.addWidget(self.spatial_view)
        spatial_note = QLabel(
            "Base：机械臂基坐标系；Pan/Tilt：云台两转轴坐标系；Camera：D435 彩色相机光学坐标系。"
            "视图按当前标定模式和当前 Pan/Tilt 角度实时更新。"
        )
        spatial_note.setWordWrap(True)
        spatial_note.setStyleSheet("color:#555;")
        spatial_layout.addWidget(spatial_note)
        left.addWidget(spatial_box, 2)
        main_area.addLayout(left, 4)

        side = QVBoxLayout()
        side.setSpacing(6)
        main_area.addLayout(side, 2)

        info_box = QGroupBox("相机信息")
        info_layout = QVBoxLayout(info_box)
        self.camera_info_label = QLabel("未连接")
        self.camera_info_label.setWordWrap(True)
        info_layout.addWidget(self.camera_info_label)
        self.performance_label = QLabel("相机输入：-- fps | 界面显示：-- fps | AprilTag：-- Hz / -- ms")
        self.performance_label.setWordWrap(True)
        self.performance_label.setStyleSheet("color:#555;")
        info_layout.addWidget(self.performance_label)
        side.addWidget(info_box)

        pan_box = QGroupBox("二自由度云台")
        pan_grid = QGridLayout(pan_box)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(100)
        self.refresh_ports_btn = QPushButton("刷新")
        self.serial_btn = QPushButton("连接")

        self.pan_spin = QSpinBox()
        self.pan_spin.setRange(0, 255)
        self.pan_spin.setReadOnly(True)
        self.pan_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.tilt_spin = QSpinBox()
        self.tilt_spin.setRange(9, 171)
        self.tilt_spin.setReadOnly(True)
        self.tilt_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 20)
        self.step_spin.setValue(2)
        self.step_spin.setToolTip("短按一次改变的云台命令量")
        self.hold_speed_spin = QSpinBox()
        self.hold_speed_spin.setRange(1, 80)
        self.hold_speed_spin.setValue(20)
        self.hold_speed_spin.setSuffix(" unit/s")
        self.hold_speed_spin.setToolTip("按钮按住超过约260 ms后，目标值连续变化的速度")
        self.tx_rate_spin = QSpinBox()
        self.tx_rate_spin.setRange(5, 20)
        self.tx_rate_spin.setValue(12)
        self.tx_rate_spin.setSuffix(" Hz")
        self.tx_rate_spin.setToolTip("串口控制帧最大发送频率；只在目标变化时发送，不会空闲持续刷包")

        self.pan_plus_btn = QPushButton("A / Pan+")
        self.pan_minus_btn = QPushButton("D / Pan−")
        self.tilt_minus_btn = QPushButton("W / Tilt−")
        self.tilt_plus_btn = QPushButton("S / Tilt+")
        self.home_btn = QPushButton("回中 (90, 90)")

        pan_grid.addWidget(QLabel("串口"), 0, 0)
        pan_grid.addWidget(self.port_combo, 0, 1)
        pan_grid.addWidget(self.refresh_ports_btn, 0, 2)
        pan_grid.addWidget(self.serial_btn, 0, 3)
        pan_grid.addWidget(QLabel("Pan Cmd"), 1, 0)
        pan_grid.addWidget(self.pan_spin, 1, 1)
        pan_grid.addWidget(QLabel("Tilt Cmd"), 1, 2)
        pan_grid.addWidget(self.tilt_spin, 1, 3)
        pan_grid.addWidget(QLabel("短按步长"), 2, 0)
        pan_grid.addWidget(self.step_spin, 2, 1)
        pan_grid.addWidget(QLabel("长按速度"), 2, 2)
        pan_grid.addWidget(self.hold_speed_spin, 2, 3)
        pan_grid.addWidget(QLabel("发送上限"), 3, 0)
        pan_grid.addWidget(self.tx_rate_spin, 3, 1)
        pan_grid.addWidget(QLabel("波特率"), 3, 2)
        baud_label = QLabel("115200")
        pan_grid.addWidget(baud_label, 3, 3)
        pan_grid.addWidget(self.pan_plus_btn, 4, 0)
        pan_grid.addWidget(self.pan_minus_btn, 4, 1)
        pan_grid.addWidget(self.tilt_minus_btn, 4, 2)
        pan_grid.addWidget(self.tilt_plus_btn, 4, 3)
        pan_grid.addWidget(self.home_btn, 5, 0, 1, 4)
        self.pantilt_control_status_label = QLabel("短按=单步；长按=连续；TX只保留最新目标")
        self.pantilt_control_status_label.setStyleSheet("color:#555; font-size:11px;")
        self.pantilt_control_status_label.setWordWrap(True)
        pan_grid.addWidget(self.pantilt_control_status_label, 6, 0, 1, 4)

        self.feedback_pan_label = QLabel("--")
        self.feedback_tilt_label = QLabel("--")
        self.feedback_pan_error_label = QLabel("--")
        self.feedback_tilt_error_label = QLabel("--")
        self.feedback_status_label = QLabel("未连接")
        self.feedback_status_label.setStyleSheet("color:#777;")
        self.feedback_usage_label = QLabel("编码器仅监视；Legacy/Base 坐标换算仍严格使用 Cmd")
        self.feedback_usage_label.setStyleSheet("color:#555; font-size:11px;")
        self.feedback_usage_label.setWordWrap(True)
        self.raw_rx_label = QLabel("--")
        self.raw_rx_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.raw_rx_label.setWordWrap(True)
        self.raw_rx_label.setToolTip("显示最近收到并按 GBK 解码的文本；悬停可查看最近原始字节 HEX。")
        pan_grid.addWidget(QLabel("Pan Enc"), 7, 0)
        pan_grid.addWidget(self.feedback_pan_label, 7, 1)
        pan_grid.addWidget(QLabel("Tilt Enc"), 7, 2)
        pan_grid.addWidget(self.feedback_tilt_label, 7, 3)
        pan_grid.addWidget(QLabel("Pan误差"), 8, 0)
        pan_grid.addWidget(self.feedback_pan_error_label, 8, 1)
        pan_grid.addWidget(QLabel("Tilt误差"), 8, 2)
        pan_grid.addWidget(self.feedback_tilt_error_label, 8, 3)
        pan_grid.addWidget(QLabel("反馈状态"), 9, 0)
        pan_grid.addWidget(self.feedback_status_label, 9, 1, 1, 3)
        pan_grid.addWidget(self.feedback_usage_label, 10, 0, 1, 4)
        pan_grid.addWidget(QLabel("串口 RX 文本"), 11, 0)
        pan_grid.addWidget(self.raw_rx_label, 11, 1, 1, 3)
        home_row = QHBoxLayout()
        self.home_pan_spin, self.home_tilt_spin = QSpinBox(), QSpinBox()
        for label, key, spin in (("Home Pan", "home_pan", self.home_pan_spin), ("Home Tilt", "home_tilt", self.home_tilt_spin)):
            spin.setRange(0, 255) if key == "home_pan" else spin.setRange(9, 171)
            spin.setValue(90)
            spin.setToolTip("回中命令值，仅影响回中按钮，不参与 Legacy 标定零位")
            home_row.addWidget(QLabel(label))
            home_row.addWidget(spin)
            spin.valueChanged.connect(lambda value, k=key: self._save_option(f"pantilt/{k}", int(value)))
            spin.valueChanged.connect(self._update_home_button_text)
        pan_grid.addLayout(home_row, 12, 0, 1, 4)
        side.addWidget(pan_box)

        detect_box = QGroupBox("识别结果")
        detect_layout = QVBoxLayout(detect_box)
        detect_layout.setSpacing(5)

        # 机器人真正使用的是 Base 坐标，因此将当前目标的 Base XYZ 独立放大显示。
        current_box = QFrame()
        current_box.setFrameShape(QFrame.Shape.StyledPanel)
        current_box.setStyleSheet(
            "QFrame { background:#f7f9fb; border:1px solid #c8d0d8; border-radius:5px; }"
            "QLabel { border:none; background:transparent; }"
        )
        current_grid = QGridLayout(current_box)
        current_grid.setContentsMargins(8, 5, 8, 5)
        current_grid.setHorizontalSpacing(10)
        current_grid.setVerticalSpacing(2)
        title = QLabel("当前目标 · 机械臂 Base")
        title.setStyleSheet("font-weight:700; font-size:14px;")
        self.current_target_id_label = QLabel("ID: --")
        self.current_target_id_label.setStyleSheet("font-weight:700; font-size:14px;")
        current_grid.addWidget(title, 0, 0, 1, 2)
        current_grid.addWidget(self.current_target_id_label, 0, 2, 1, 2)
        self.current_base_x_label = QLabel("X = -- mm")
        self.current_base_y_label = QLabel("Y = -- mm")
        self.current_base_z_label = QLabel("Z = -- mm")
        for lab in (self.current_base_x_label, self.current_base_y_label, self.current_base_z_label):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet("font-weight:700; font-size:17px; padding:3px 8px;")
        current_grid.addWidget(self.current_base_x_label, 1, 0)
        current_grid.addWidget(self.current_base_y_label, 1, 1)
        current_grid.addWidget(self.current_base_z_label, 1, 2)
        self.current_target_hint = QLabel("单码自动显示；多码时点击下方结果表的任意一行切换主显示")
        self.current_target_hint.setStyleSheet("color:#666;")
        current_grid.addWidget(self.current_target_hint, 2, 0, 1, 4)
        detect_layout.addWidget(current_box)

        # Base 坐标优先、单位 mm；相机坐标保留用于调试。
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels([
            "ID",
            "目标点 @ Base\nXYZ (mm)",
            "目标点 @ Camera\nXYZ (mm)",
            "Tag中心 @ Camera\nXYZ (mm)",
        ])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.result_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.result_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.result_table.setWordWrap(False)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setMinimumHeight(155)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(85)
        header.setToolTip(
            "目标点@Base：实际给机械臂使用的目标点坐标，单位mm；"
            "目标点@Camera与Tag中心@Camera用于视觉/坐标变换调试。"
        )
        self.result_table.itemSelectionChanged.connect(self._on_result_selection_changed)
        detect_layout.addWidget(self.result_table)
        side.insertWidget(0, detect_box, 1)

        note = QLabel(
            "坐标定义：Camera 光学坐标 +X 向图像右、+Y 向图像下、+Z 向镜头前方；"
            "Base XYZ 完全遵循机械臂自身基坐标系。手动 Base→Pan 的 XYZ 是‘Pan 轴中心在 Base 坐标系中的坐标’；"
            "Pan→Tilt 的 XYZ 是‘中立位时 Tilt 轴中心相对 Pan 轴中心、用 Pan 坐标系表达的偏移’，因此会随 Pan 一起旋转。"
            "云台→机械臂可在‘标定文件’和‘手动机械参数’两种方式间切换。"
            "界面参数及标定编辑副本写入 settings.ini；相机参数成功写入硬件后保存。编码器反馈已接入并用于监视 Cmd/Enc 误差；"
            "为保持旧联合标定语义，当前 Base 坐标换算仍严格使用 Pan/Tilt Cmd。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555;")
        side.addWidget(note)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(105)
        log_layout.addWidget(self.log_edit)
        root.addWidget(log_box)

        self.statusBar().showMessage("就绪")

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.refresh_btn.clicked.connect(lambda: self.refresh_profiles(show_dialog=True))
        self.tag_size_spin.valueChanged.connect(self.on_tag_size_changed)
        self.median_box.toggled.connect(self.on_median_changed)
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        self.detect_hz_spin.valueChanged.connect(self.on_detect_hz_changed)
        self.preview_mode_combo.currentIndexChanged.connect(self.on_preview_mode_changed)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen_preview)
        self.offset_select_btn.clicked.connect(self.select_or_add_offset_id)
        self.offset_delete_btn.clicked.connect(self.delete_current_offset_id)
        self.offset_id_combo.activated.connect(self.on_offset_id_combo_activated)
        self.offset_table.cellClicked.connect(self.on_offset_table_clicked)

        self.refresh_ports_btn.clicked.connect(self.refresh_serial_ports)
        self.serial_btn.clicked.connect(self.toggle_serial)
        # v1.7: short press performs one step; holding starts continuous motion.
        # Serial writes are not performed in these button callbacks.
        self.pan_plus_btn.pressed.connect(lambda: self.start_pantilt_hold(+1, 0))
        self.pan_minus_btn.pressed.connect(lambda: self.start_pantilt_hold(-1, 0))
        self.tilt_minus_btn.pressed.connect(lambda: self.start_pantilt_hold(0, -1))
        self.tilt_plus_btn.pressed.connect(lambda: self.start_pantilt_hold(0, +1))
        for _btn in (self.pan_plus_btn, self.pan_minus_btn, self.tilt_minus_btn, self.tilt_plus_btn):
            _btn.released.connect(self.stop_pantilt_hold)
        self.home_btn.clicked.connect(self.home_pantilt)
        self.step_spin.valueChanged.connect(self.on_pantilt_step_changed)
        self.hold_speed_spin.valueChanged.connect(self.on_pantilt_hold_speed_changed)
        self.tx_rate_spin.valueChanged.connect(self.on_pantilt_tx_rate_changed)
        self.spatial_view.view_changed.connect(self.on_spatial_view_changed)

    def _make_offset_spin(self):
        spin = QDoubleSpinBox()
        spin.setRange(-5000.0, 5000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" mm")
        spin.setMinimumWidth(125)
        spin.setToolTip("当前所选 AprilTag ID 的目标点物理偏移，单位 mm")
        spin.valueChanged.connect(self.on_current_id_offset_changed)
        return spin

    def _restore_saved_options(self):
        """Restore all user-adjustable options saved from the previous run."""
        try:
            tag_size = float(self.settings.value("detection/tag_size_mm", 100.0))
        except Exception:
            tag_size = 100.0
        self.tag_size_spin.blockSignals(True)
        self.tag_size_spin.setValue(tag_size)
        self.tag_size_spin.blockSignals(False)

        median = self.settings.value("detection/median_enabled", True, type=bool)
        self.median_box.blockSignals(True)
        self.median_box.setChecked(bool(median))
        self.median_box.blockSignals(False)

        try:
            detect_hz = float(self.settings.value("detection/target_hz", 15.0))
        except Exception:
            detect_hz = 15.0
        self.detect_hz_spin.blockSignals(True)
        self.detect_hz_spin.setValue(max(1.0, min(30.0, detect_hz)))
        self.detect_hz_spin.blockSignals(False)

        preview_mode = self.settings.value("camera/preview_mode", "fit", type=str)
        preview_idx = self.preview_mode_combo.findData(preview_mode)
        self.preview_mode_combo.blockSignals(True)
        self.preview_mode_combo.setCurrentIndex(preview_idx if preview_idx >= 0 else 0)
        self.preview_mode_combo.blockSignals(False)
        self._apply_preview_mode()

        self.tag_offsets_mm = self._load_tag_offsets()
        self._refresh_offset_editor()

        for key, spin in (("home_pan", self.home_pan_spin), ("home_tilt", self.home_tilt_spin)):
            try:
                value = int(self.settings.value(f"pantilt/{key}", 90))
            except (TypeError, ValueError):
                value = 90
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        try:
            view_yaw = float(self.settings.value("view3d/yaw_deg", -38.0))
            view_pitch = float(self.settings.value("view3d/pitch_deg", 24.0))
            view_zoom = float(self.settings.value("view3d/zoom", 1.0))
            self.spatial_view.set_view(view_yaw, view_pitch, view_zoom)
        except Exception:
            self.spatial_view.set_view(-38.0, 24.0, 1.0)
        self._update_home_button_text()

        # Restore pan/tilt serial settings and the last commanded software position.
        saved_port = self.settings.value("pantilt/serial_port", "COM17", type=str)
        try:
            saved_pan = int(self.settings.value("pantilt/pan", 90))
        except Exception:
            saved_pan = 90
        try:
            saved_tilt = int(self.settings.value("pantilt/tilt", 90))
        except Exception:
            saved_tilt = 90
        try:
            saved_step = int(self.settings.value("pantilt/step", 2))
        except Exception:
            saved_step = 2
        try:
            saved_hold_speed = int(self.settings.value("pantilt/hold_speed", 20))
        except Exception:
            saved_hold_speed = 20
        try:
            saved_tx_rate = int(self.settings.value("pantilt/tx_rate_hz", 12))
        except Exception:
            saved_tx_rate = 12

        self.pan_spin.setValue(max(0, min(255, saved_pan)))
        self.tilt_spin.setValue(max(9, min(171, saved_tilt)))
        self.step_spin.blockSignals(True)
        self.step_spin.setValue(max(1, min(20, saved_step)))
        self.step_spin.blockSignals(False)
        self.hold_speed_spin.blockSignals(True)
        self.hold_speed_spin.setValue(max(1, min(80, saved_hold_speed)))
        self.hold_speed_spin.blockSignals(False)
        self.tx_rate_spin.blockSignals(True)
        self.tx_rate_spin.setValue(max(5, min(20, saved_tx_rate)))
        self.tx_rate_spin.blockSignals(False)
        self.pantilt.set_tx_rate_hz(self.tx_rate_spin.value())
        self.port_combo.clear()
        if saved_port:
            self.port_combo.addItem(saved_port)
            self.port_combo.setCurrentText(saved_port)
        self.refresh_serial_ports(show_log=False)

    def _update_home_button_text(self):
        self.home_btn.setText(f"回中 ({self.home_pan_spin.value()}, {self.home_tilt_spin.value()})")

    def _load_saved_calibrations(self):
        self.calibration_panel.restore()

    def on_calibration_chain_changed(self, chain):
        self.calibration = chain
        source = "文件原值" if chain.active_source == "original" else "编辑副本"
        self.active_source_status.setText("生效：" + source + (" · Base停用" if chain.blocked_reason else " · Cmd"))
        self._update_table(self.last_results)

    def request_rgb_options(self, values):
        worker = self.camera_worker
        if worker and worker.isRunning():
            generation = worker.request_options(values)
            if generation is not None:
                self.rgb_options_panel.minimum_generation = generation

    def on_rgb_options_snapshot(self, payload):
        if self.camera_worker and self.camera_worker.isRunning() and not self._camera_stopping:
            self.rgb_options_panel.update_snapshot(payload)

    def _angles_for_transform(self):
        # v1.6 policy: keep the existing calibration semantics exactly unchanged.
        # The old camera-pan/tilt and arm-pan/tilt calibrations were fitted in
        # command-value space, so encoder feedback is intentionally NOT fed into
        # the Legacy/Base transform yet.  Encoder values are monitoring data only.
        return float(self.pan_spin.value()), float(self.tilt_spin.value()), "command_legacy"

    def poll_pantilt_rx(self):
        """Refresh verified encoder feedback parsed by the background RX thread."""
        if not self.pantilt.connected:
            # If the receive thread lost the port unexpectedly, surface that state
            # instead of silently leaving the UI looking connected.
            if self.serial_btn.text() == "断开":
                snap = self.pantilt.feedback_snapshot(stale_after_s=1.0)
                tx = self.pantilt.tx_snapshot()
                err = str(tx.get("error") or snap.get("error") or "").strip()
                self.feedback_valid = False
                self._feedback_online_prev = False
                self.feedback_status_label.setText("串口连接已丢失" + (f"：{err}" if err else ""))
                self.feedback_status_label.setStyleSheet("color:#b00020; font-weight:600;")
                self.serial_btn.setText("连接")
                self.port_combo.setEnabled(True)
                self.refresh_ports_btn.setEnabled(True)
                if err:
                    self._append_log("云台串口后台读取已停止：" + err)
            return

        tx = self.pantilt.tx_snapshot()
        tx_err = str(tx.get("error") or "").strip()
        if tx_err and tx_err != self._pantilt_last_tx_error:
            self._pantilt_last_tx_error = tx_err
            self._append_log("云台串口后台发送异常：" + tx_err)
        elif not tx_err:
            self._pantilt_last_tx_error = ""
        if tx_err:
            self.pantilt_control_status_label.setText("TX异常：" + tx_err)
            self.pantilt_control_status_label.setStyleSheet("color:#b00020; font-weight:600;")
        else:
            pending_text = " · 待发送最新目标" if tx.get("pending") else ""
            self.pantilt_control_status_label.setStyleSheet("color:#555; font-size:11px;")
            if self._pantilt_hold_started:
                self.pantilt_control_status_label.setText(
                    f"连续运动中 · {self.hold_speed_spin.value()} unit/s · TX≤{self.tx_rate_spin.value()} Hz{pending_text}"
                )
            elif self._pantilt_hold_dir == (0, 0):
                self.pantilt_control_status_label.setText(
                    f"短按=单步；长按=连续 · TX≤{self.tx_rate_spin.value()} Hz · 最新目标覆盖{pending_text}"
                )

        snap = self.pantilt.feedback_snapshot(stale_after_s=1.0)
        if snap.get("error"):
            self.feedback_status_label.setText("串口读取异常：" + str(snap["error"]))
            self.feedback_status_label.setStyleSheet("color:#b00020; font-weight:600;")
            return

        text_tail = str(snap.get("text_tail") or "").replace("\r", " ").replace("\n", "  ").strip()
        if text_tail:
            self.raw_rx_label.setText(text_tail[-180:])
        else:
            self.raw_rx_label.setText("--")
        self.raw_rx_label.setToolTip(str(snap.get("raw_hex") or ""))

        pan = snap.get("pan")
        tilt = snap.get("tilt")
        online = bool(snap.get("online"))
        self.feedback_valid = online
        self.feedback_pan = None if pan is None else float(pan)
        self.feedback_tilt = None if tilt is None else float(tilt)

        self.feedback_pan_label.setText("--" if pan is None else f"{float(pan):.1f}°")
        self.feedback_tilt_label.setText("--" if tilt is None else f"{float(tilt):.1f}°")

        if pan is not None:
            ep = float(pan) - float(self.pan_spin.value())
            self.feedback_pan_error_label.setText(f"{ep:+.1f}°")
        else:
            self.feedback_pan_error_label.setText("--")
        if tilt is not None:
            et = float(tilt) - float(self.tilt_spin.value())
            self.feedback_tilt_error_label.setText(f"{et:+.1f}°")
        else:
            self.feedback_tilt_error_label.setText("--")

        if online:
            age_ms = 1000.0 * float(snap.get("age_s") or 0.0)
            self.feedback_status_label.setText(f"在线 · 最近完整 A/B 更新 {age_ms:.0f} ms 前 · GBK主动回传")
            self.feedback_status_label.setStyleSheet("color:#187a2f; font-weight:600;")
            if not self._feedback_online_prev:
                self._append_log(
                    f"编码器反馈已在线：Pan Enc={float(pan):.1f}°, Tilt Enc={float(tilt):.1f}°；"
                    "Base坐标仍按Legacy Cmd计算。"
                )
        else:
            age = snap.get("age_s")
            last_rx_age = snap.get("last_rx_age_s")
            if age is not None:
                self.feedback_status_label.setText(f"反馈超时 · 最后完整 A/B 为 {float(age):.2f} s 前")
                self.feedback_status_label.setStyleSheet("color:#b00020; font-weight:600;")
            elif last_rx_age is not None:
                self.feedback_status_label.setText("已收到串口数据，等待完整的舵机A/B角度反馈…")
                self.feedback_status_label.setStyleSheet("color:#a65b00;")
            else:
                self.feedback_status_label.setText("已连接，等待编码器主动反馈…")
                self.feedback_status_label.setStyleSheet("color:#a65b00;")
            if self._feedback_online_prev:
                self._append_log("编码器反馈已超时；Base坐标计算不受影响，仍严格使用 Pan/Tilt Cmd。")

        self._feedback_online_prev = online

    @pyqtSlot(float, float, float)
    def on_spatial_view_changed(self, yaw_deg, pitch_deg, zoom):
        # Persist the interactive 3-D view immediately, just like the other editable options.
        self._save_option("view3d/yaw_deg", float(yaw_deg))
        self._save_option("view3d/pitch_deg", float(pitch_deg))
        self._save_option("view3d/zoom", float(zoom))

    def _save_option(self, key, value):
        self.settings.setValue(key, value)
        self.settings.sync()

    @pyqtSlot()
    def refresh_profiles(self, show_dialog=True):
        if self._camera_running():
            if show_dialog:
                QMessageBox.information(self, "提示", "请先停止相机，再刷新分辨率。")
            return

        old_data = self.profile_combo.currentData()
        saved_profile_key = self.settings.value("camera/profile_key", "", type=str)
        self._loading_profiles = True
        self.profile_combo.clear()

        try:
            name, serial, profiles = discover_color_profiles()
        except Exception as e:
            traceback.print_exc()
            name, serial, profiles = None, None, []
            self._append_log(f"读取相机分辨率失败：{e}")

        if profiles:
            for w, h, fps, fmt in profiles:
                text = f"{w}×{h} @ {fps} fps  [{profile_format_name(fmt)}]"
                self.profile_combo.addItem(text, (w, h, fps, fmt))

            # Priority: saved choice -> current in-session choice -> highest resolution.
            selected_idx = -1
            if saved_profile_key:
                for i in range(self.profile_combo.count()):
                    if profile_key(self.profile_combo.itemData(i)) == saved_profile_key:
                        selected_idx = i
                        break

            if selected_idx < 0 and old_data is not None:
                old_key = profile_key(old_data)
                for i in range(self.profile_combo.count()):
                    if profile_key(self.profile_combo.itemData(i)) == old_key:
                        selected_idx = i
                        break

            # discover_color_profiles() is sorted highest-resolution first.
            if selected_idx < 0:
                selected_idx = 0

            self.profile_combo.setCurrentIndex(selected_idx)
            selected = self.profile_combo.currentData()
            if selected:
                self._save_option("camera/profile_key", profile_key(selected))

            self._append_log(
                f"已读取 {name} S/N {serial}：发现 {len(profiles)} 个可用彩色流配置。"
            )
            self._append_log(
                f"当前默认彩色流：{self.profile_combo.currentText()}（首次运行默认选择最高分辨率；以后恢复上次选择）。"
            )
        else:
            # Fallback entries allow the GUI to remain usable even before the camera is connected.
            fallback_fmt = _rs_format("yuyv") or _rs_format("bgr8") or _rs_format("rgb8")
            fallback = [
                (1920, 1080, 30, fallback_fmt),
                (1280, 720, 30, fallback_fmt),
                (640, 480, 30, fallback_fmt),
            ]
            for w, h, fps, fmt in fallback:
                self.profile_combo.addItem(f"{w}×{h} @ {fps} fps [{profile_format_name(fmt)}] (fallback)", (w, h, fps, fmt))
            selected_idx = 0
            if saved_profile_key:
                for i in range(self.profile_combo.count()):
                    if profile_key(self.profile_combo.itemData(i)) == saved_profile_key:
                        selected_idx = i
                        break
            self.profile_combo.setCurrentIndex(selected_idx)
            selected = self.profile_combo.currentData()
            if selected:
                self._save_option("camera/profile_key", profile_key(selected))
            self._append_log("未读取到 RealSense 彩色配置，已显示备用分辨率。")
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "未读取到相机",
                    "没有读取到 RealSense 的可用彩色流配置。\n"
                    "请确认相机已连接，然后点击‘刷新分辨率’。"
                )

        self._loading_profiles = False

    @pyqtSlot()
    def refresh_serial_ports(self, show_log=True):
        current = self.port_combo.currentText().strip() or self.settings.value(
            "pantilt/serial_port", "COM17", type=str
        )
        ports = PantiltSerial.list_ports()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        if current and self.port_combo.findText(current) < 0:
            self.port_combo.addItem(current)
        if current:
            self.port_combo.setCurrentText(current)
        self.port_combo.blockSignals(False)
        if show_log:
            if ports:
                self._append_log("串口列表已刷新：" + ", ".join(ports))
            else:
                self._append_log("未枚举到串口；仍可在串口框中手动输入 COM 号。")

    @pyqtSlot()
    def toggle_serial(self):
        if self.pantilt.connected:
            self.stop_pantilt_hold(save=True, log=False)
            self.pantilt.disconnect()
            self.feedback_valid = False
            self.feedback_pan = None
            self.feedback_tilt = None
            self._feedback_online_prev = False
            self.feedback_pan_label.setText("--")
            self.feedback_tilt_label.setText("--")
            self.feedback_pan_error_label.setText("--")
            self.feedback_tilt_error_label.setText("--")
            self.feedback_status_label.setText("串口已断开")
            self.feedback_status_label.setStyleSheet("color:#777;")
            self.raw_rx_label.setText("--")
            self.raw_rx_label.setToolTip("")
            self.serial_btn.setText("连接")
            self.port_combo.setEnabled(True)
            self.refresh_ports_btn.setEnabled(True)
            self._update_table(self.last_results)
            self._append_log("云台串口已断开。")
            return

        try:
            port = self.port_combo.currentText().strip()
            self.pantilt.connect(port, 115200)
            self.raw_rx_label.setText("--")
            self.raw_rx_label.setToolTip("")
            self.feedback_valid = False
            self.feedback_pan = None
            self.feedback_tilt = None
            self._feedback_online_prev = False
            self.feedback_pan_label.setText("--")
            self.feedback_tilt_label.setText("--")
            self.feedback_pan_error_label.setText("--")
            self.feedback_tilt_error_label.setText("--")
            self.feedback_status_label.setText("已连接，等待编码器主动反馈…")
            self.feedback_status_label.setStyleSheet("color:#a65b00;")
            # Synchronize the controller's software record only; do not move on connect.
            self.pantilt.pan = self.pan_spin.value()
            self.pantilt.tilt = self.tilt_spin.value()
            self.pantilt.set_tx_rate_hz(self.tx_rate_spin.value())
            self.serial_btn.setText("断开")
            self.port_combo.setEnabled(False)
            self.refresh_ports_btn.setEnabled(False)
            self._save_option("pantilt/serial_port", port)
            self._append_log(
                f"云台已连接 {port} @ 115200 8N1；后台开始解析 GBK 编码器主动反馈。"
                "连接动作本身不会驱动云台，Legacy Base坐标仍严格使用 Cmd。"
            )
        except PantiltError as exc:
            QMessageBox.warning(self, "云台连接失败", str(exc))
            self._append_log(str(exc))

    def command_pantilt(self, pan: int, tilt: int, *, persist: bool = True, log: bool = True, update_results: bool = True):
        """Set the latest command target; actual serial TX is asynchronous."""
        try:
            pan, tilt = self.pantilt.command(pan, tilt)
            self.pan_spin.setValue(pan)
            self.tilt_spin.setValue(tilt)
            if persist:
                self._save_option("pantilt/pan", int(pan))
                self._save_option("pantilt/tilt", int(tilt))
            if log:
                self._append_log(f"云台目标更新 pan={pan}, tilt={tilt}（最新目标覆盖，异步限频发送）")
            if update_results:
                self._update_table(self.last_results)
            return True
        except PantiltError as exc:
            # A press on a disconnected controller should fail immediately.
            if log:
                QMessageBox.warning(self, "云台控制失败", str(exc))
                self._append_log(str(exc))
            if not self.pantilt.connected:
                self.serial_btn.setText("连接")
                self.port_combo.setEnabled(True)
                self.refresh_ports_btn.setEnabled(True)
            return False

    def jog_pantilt(self, dp: int, dt: int):
        return self.command_pantilt(self.pan_spin.value() + int(dp), self.tilt_spin.value() + int(dt))

    def start_pantilt_hold(self, pan_dir: int, tilt_dir: int):
        """One immediate step, then continuous motion if the button stays held."""
        if not self.pantilt.connected:
            self.command_pantilt(self.pan_spin.value(), self.tilt_spin.value(), log=True)
            return
        self.stop_pantilt_hold(save=False, log=False)
        self._pantilt_hold_dir = (int(pan_dir), int(tilt_dir))
        self._pantilt_hold_started = False

        step = int(self.step_spin.value())
        ok = self.command_pantilt(
            self.pan_spin.value() + int(pan_dir) * step,
            self.tilt_spin.value() + int(tilt_dir) * step,
            persist=False,
            log=False,
            update_results=True,
        )
        if not ok:
            self._pantilt_hold_dir = (0, 0)
            return
        self._pantilt_hold_float_pan = float(self.pan_spin.value())
        self._pantilt_hold_float_tilt = float(self.tilt_spin.value())
        self.pantilt_hold_delay_timer.start()
        self.pantilt_control_status_label.setText(
            f"已点动；继续按住将连续运动 · TX≤{self.tx_rate_spin.value()} Hz · 最新目标覆盖"
        )

    def _begin_pantilt_hold_repeat(self):
        if self._pantilt_hold_dir == (0, 0) or not self.pantilt.connected:
            return
        self._pantilt_hold_started = True
        self._pantilt_hold_last_t = time.monotonic()
        self.pantilt_hold_repeat_timer.start()
        self.pantilt_control_status_label.setText(
            f"连续运动中 · {self.hold_speed_spin.value()} unit/s · TX≤{self.tx_rate_spin.value()} Hz · 松开即停"
        )
        self._append_log(
            f"云台开始连续运动 dir={self._pantilt_hold_dir}，速度={self.hold_speed_spin.value()} unit/s，"
            f"发送上限={self.tx_rate_spin.value()} Hz。"
        )

    def _pantilt_hold_tick(self):
        if self._pantilt_hold_dir == (0, 0) or not self.pantilt.connected:
            self.stop_pantilt_hold()
            return
        now = time.monotonic()
        dt = max(0.0, min(0.15, now - self._pantilt_hold_last_t))
        self._pantilt_hold_last_t = now
        speed = float(self.hold_speed_spin.value())
        dp, dt_dir = self._pantilt_hold_dir
        self._pantilt_hold_float_pan += float(dp) * speed * dt
        self._pantilt_hold_float_tilt += float(dt_dir) * speed * dt
        self._pantilt_hold_float_pan = max(0.0, min(255.0, self._pantilt_hold_float_pan))
        self._pantilt_hold_float_tilt = max(9.0, min(171.0, self._pantilt_hold_float_tilt))
        pan = int(round(self._pantilt_hold_float_pan))
        tilt = int(round(self._pantilt_hold_float_tilt))

        # Do not log, sync settings.ini or create a queue for every 50-ms tick.
        # PantiltSerial keeps only this newest requested target.
        self.command_pantilt(pan, tilt, persist=False, log=False, update_results=False)

        # Stop advancing a saturated axis instead of repeatedly requesting the
        # same hard-limit target forever.
        hit_pan_limit = dp != 0 and ((dp > 0 and pan >= 255) or (dp < 0 and pan <= 0))
        hit_tilt_limit = dt_dir != 0 and ((dt_dir > 0 and tilt >= 171) or (dt_dir < 0 and tilt <= 9))
        if hit_pan_limit or hit_tilt_limit:
            self.stop_pantilt_hold()

    def stop_pantilt_hold(self, *, save: bool = True, log: bool = True):
        if hasattr(self, "pantilt_hold_delay_timer"):
            self.pantilt_hold_delay_timer.stop()
        if hasattr(self, "pantilt_hold_repeat_timer"):
            self.pantilt_hold_repeat_timer.stop()
        was_active = self._pantilt_hold_dir != (0, 0)
        was_continuous = self._pantilt_hold_started
        self._pantilt_hold_dir = (0, 0)
        self._pantilt_hold_started = False
        if save and was_active:
            self._save_option("pantilt/pan", int(self.pan_spin.value()))
            self._save_option("pantilt/tilt", int(self.tilt_spin.value()))
            self._update_table(self.last_results)
        if log and was_active:
            mode = "连续运动停止" if was_continuous else "单步点动完成"
            self._append_log(f"云台{mode}：pan={self.pan_spin.value()}, tilt={self.tilt_spin.value()}")
        if hasattr(self, "pantilt_control_status_label"):
            self.pantilt_control_status_label.setText(
                f"短按=单步；长按=连续 · TX≤{self.tx_rate_spin.value()} Hz · 最新目标覆盖"
            )

    @pyqtSlot()
    def home_pantilt(self):
        self.stop_pantilt_hold(save=False, log=False)
        self.command_pantilt(
            int(round(self.home_pan_spin.value())),
            int(round(self.home_tilt_spin.value())),
        )

    @pyqtSlot(int)
    def on_pantilt_step_changed(self, value):
        self._save_option("pantilt/step", int(value))

    @pyqtSlot(int)
    def on_pantilt_hold_speed_changed(self, value):
        self._save_option("pantilt/hold_speed", int(value))

    @pyqtSlot(int)
    def on_pantilt_tx_rate_changed(self, value):
        hz = int(value)
        self.pantilt.set_tx_rate_hz(hz)
        self._save_option("pantilt/tx_rate_hz", hz)
        if hasattr(self, "pantilt_control_status_label"):
            self.pantilt_control_status_label.setText(
                f"短按=单步；长按=连续 · TX≤{hz} Hz · 最新目标覆盖"
            )

    def _camera_running(self):
        return bool(self.camera_worker and self.camera_worker.isRunning())

    @pyqtSlot()
    def start_camera(self):
        if self._camera_running() or (self.detector_worker and self.detector_worker.isRunning()):
            return

        data = self.profile_combo.currentData()
        if not data:
            QMessageBox.warning(self, "分辨率错误", "当前没有可用的相机分辨率。")
            return

        width, height, fps, pixel_format = data
        self.frame_buffer.clear()
        self._latest_detection_captured_at = 0.0
        self.latest_overlays = []
        self.last_results = []
        self._latest_detection_received = 0.0
        self._latest_detection_frame_seq = -1
        self._last_preview_seq = -1
        self._capture_fps_value = 0.0
        self._preview_fps_value = 0.0
        self._detection_fps_value = 0.0
        self._detection_ms_value = 0.0
        self._preview_fps_t0 = time.monotonic()
        self._preview_fps_frames = 0

        self.camera_worker = CameraCaptureWorker(
            self.frame_buffer,
            width=width,
            height=height,
            fps=fps,
            pixel_format=pixel_format,
            parent=self,
        )
        self.detector_worker = DetectionWorker(
            self.frame_buffer,
            tag_size_mm=self.tag_size_spin.value(),
            median_enabled=self.median_box.isChecked(),
            target_offsets_mm=self.tag_offsets_mm,
            target_hz=self.detect_hz_spin.value(),
            detect_max_width=960,
            parent=self,
        )

        self.camera_worker.status.connect(self.on_status)
        self.camera_worker.error.connect(self.on_error)
        self.camera_worker.camera_info.connect(self.on_camera_info)
        self.camera_worker.stats.connect(self.on_capture_stats)
        self.camera_worker.rgb_options.connect(self.on_rgb_options_snapshot)
        self.camera_worker.finished.connect(self.on_camera_worker_finished)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)

        self.detector_worker.status.connect(self.on_status)
        self.detector_worker.error.connect(self.on_detection_error)
        self.detector_worker.detection_ready.connect(self.on_detection_ready)
        self.detector_worker.finished.connect(self.on_detector_worker_finished)
        self.detector_worker.finished.connect(self.detector_worker.deleteLater)

        self.tag_size_changed.connect(self.detector_worker.set_tag_size_mm)
        self.median_changed.connect(self.detector_worker.set_median_enabled)
        self.target_offsets_changed.connect(self.detector_worker.set_target_offsets_mm)

        self._camera_stopping = False
        self.rgb_options_panel.reset()
        self.camera_worker.start()
        self.detector_worker.start()
        self.preview_timer.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.profile_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self._append_log(
            f"正在启动彩色流 {width}×{height}@{fps} {profile_format_name(pixel_format)}；"
            f"AprilTag目标 {self.detect_hz_spin.value():.0f} Hz，预览只取最新帧。"
        )

    @pyqtSlot()
    def stop_camera(self):
        self._camera_stopping = True
        self._latest_detection_captured_at = 0.0
        self.results_updated.emit(None)
        self.preview_timer.stop()
        self.rgb_options_panel.reset()
        self.stop_btn.setEnabled(False)
        if self.detector_worker and self.detector_worker.isRunning():
            self.detector_worker.stop()
        if self.camera_worker and self.camera_worker.isRunning():
            self._append_log("正在停止采集及参数线程…")
            self.camera_worker.stop()
        else:
            self.camera_worker = None
            self.detector_worker = None
            self._set_stopped_ui()
        self.frame_buffer.clear()

    @pyqtSlot(float)
    def on_tag_size_changed(self, value):
        self._save_option("detection/tag_size_mm", float(value))
        self.tag_size_changed.emit(float(value))
        self._append_log(f"Tag 实际边长已改为 {value:.1f} mm，并已保存为下次默认值。")

    @pyqtSlot(bool)
    def on_median_changed(self, checked):
        self._save_option("detection/median_enabled", bool(checked))
        self.median_changed.emit(bool(checked))
        self._append_log(f"5帧中值稳定：{'开启' if checked else '关闭'}，并已保存。")

    @pyqtSlot(float)
    def on_detect_hz_changed(self, value):
        value = max(1.0, min(30.0, float(value)))
        self._save_option("detection/target_hz", value)
        if self.detector_worker is not None:
            self.detector_worker.set_target_hz(value)
        self._update_performance_label()

    @pyqtSlot(int)
    def on_preview_mode_changed(self, _index):
        mode = self.preview_mode_combo.currentData() or "fit"
        self._save_option("camera/preview_mode", mode)
        self._apply_preview_mode()
        self._last_preview_seq = -1

    def _apply_preview_mode(self):
        if not hasattr(self, "video_scroll"):
            return
        mode = self.preview_mode_combo.currentData() or "fit"
        if mode == "1to1":
            self.video_scroll.setWidgetResizable(False)
            seq, _ts, frame, _k, _d, meta = self.frame_buffer.snapshot() if hasattr(self, "frame_buffer") else (0,0,None,None,None,{})
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                w = int(meta.get("width", DEFAULT_WIDTH)) if meta else DEFAULT_WIDTH
                h = int(meta.get("height", DEFAULT_HEIGHT)) if meta else DEFAULT_HEIGHT
            self.video_label.setMinimumSize(w, h)
            self.video_label.setMaximumSize(w, h)
            self.video_label.resize(w, h)
            self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        else:
            self.video_scroll.setWidgetResizable(True)
            self.video_label.setMinimumSize(1, 1)
            self.video_label.setMaximumSize(16777215, 16777215)
            self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    @pyqtSlot()
    def toggle_fullscreen_preview(self):
        if self.fullscreen_preview is not None and self.fullscreen_preview.isVisible():
            self.fullscreen_preview.close()
            return
        self.fullscreen_preview = FullscreenPreviewWindow()
        self.fullscreen_preview.closed.connect(self._on_fullscreen_closed)
        self.fullscreen_preview.showFullScreen()
        self.fullscreen_btn.setText("退出全屏")
        self._last_preview_seq = -1

    @pyqtSlot()
    def _on_fullscreen_closed(self):
        self.fullscreen_btn.setText("全屏预览")
        if self.fullscreen_preview is not None:
            self.fullscreen_preview.deleteLater()
        self.fullscreen_preview = None
        self._last_preview_seq = -1

    def _load_tag_offsets(self):
        raw = self.settings.value("target/per_id_offsets_json", "", type=str)
        mapping = {}
        if raw:
            try:
                data = json.loads(raw)
                for key, value in dict(data).items():
                    tag_id = int(key)
                    if not 0 <= tag_id <= 586:
                        continue
                    vals = [float(v) for v in value[:3]]
                    if len(vals) == 3 and all(np.isfinite(vals)):
                        mapping[tag_id] = vals
            except Exception as exc:
                self._append_log(f"读取每ID偏移配置失败，已使用空配置：{exc}")
        return mapping

    def _save_tag_offsets(self):
        serializable = {str(int(k)): [float(v) for v in vals[:3]]
                        for k, vals in sorted(self.tag_offsets_mm.items())}
        self.settings.setValue("target/per_id_offsets_json", json.dumps(serializable, ensure_ascii=False, separators=(",", ":")))
        self.settings.sync()
        self.target_offsets_changed.emit(dict(serializable))

    def _parse_offset_id(self):
        text = self.offset_id_combo.currentText().strip()
        try:
            tag_id = int(text)
        except Exception:
            return None
        if not 0 <= tag_id <= 586:
            return None
        return tag_id

    def _set_offset_spin_values(self, vals):
        for spin, value in zip((self.offset_x_spin, self.offset_y_spin, self.offset_z_spin), vals):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _ensure_id_choice(self, tag_id):
        text = str(int(tag_id))
        if self.offset_id_combo.findText(text) < 0:
            self.offset_id_combo.addItem(text, int(tag_id))

    def _refresh_offset_editor(self, select_id=None):
        current_text = self.offset_id_combo.currentText().strip() if hasattr(self, "offset_id_combo") else ""
        detected = getattr(self, "detected_tag_ids", set())
        ids = sorted(set(self.tag_offsets_mm.keys()) | set(detected))
        self.offset_id_combo.blockSignals(True)
        self.offset_id_combo.clear()
        for tag_id in ids:
            self.offset_id_combo.addItem(str(tag_id), tag_id)
        if select_id is None:
            try:
                select_id = int(current_text)
            except Exception:
                select_id = ids[0] if ids else 0
        self.offset_id_combo.setCurrentText(str(select_id))
        self.offset_id_combo.blockSignals(False)

        self.offset_table.setRowCount(len(self.tag_offsets_mm))
        for row, tag_id in enumerate(sorted(self.tag_offsets_mm)):
            vals = self.tag_offsets_mm[tag_id]
            for col, text in enumerate((str(tag_id), f"{vals[0]:+.1f}", f"{vals[1]:+.1f}", f"{vals[2]:+.1f}")):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.offset_table.setItem(row, col, item)

        vals = self.tag_offsets_mm.get(int(select_id), [0.0, 0.0, 0.0])
        self._set_offset_spin_values(vals)

    @pyqtSlot()
    def select_or_add_offset_id(self):
        tag_id = self._parse_offset_id()
        if tag_id is None:
            QMessageBox.warning(self, "ID无效", "请输入 AprilTag 36h11 的有效 ID（0~586）。")
            return
        if tag_id not in self.tag_offsets_mm:
            self.tag_offsets_mm[tag_id] = [0.0, 0.0, 0.0]
            self._save_tag_offsets()
            self._append_log(f"已新增 ID {tag_id} 偏移配置：X=0, Y=0, Z=0 mm")
        self._refresh_offset_editor(select_id=tag_id)

    @pyqtSlot(int)
    def on_offset_id_combo_activated(self, _index):
        tag_id = self._parse_offset_id()
        if tag_id is None:
            return
        vals = self.tag_offsets_mm.get(tag_id, [0.0, 0.0, 0.0])
        self._set_offset_spin_values(vals)

    @pyqtSlot(int, int)
    def on_offset_table_clicked(self, row, _col):
        item = self.offset_table.item(row, 0)
        if item is None:
            return
        try:
            tag_id = int(item.text())
        except Exception:
            return
        self.offset_id_combo.setCurrentText(str(tag_id))
        self._set_offset_spin_values(self.tag_offsets_mm.get(tag_id, [0.0, 0.0, 0.0]))

    @pyqtSlot(float)
    def on_current_id_offset_changed(self, _value):
        tag_id = self._parse_offset_id()
        if tag_id is None:
            return
        vals = [float(self.offset_x_spin.value()), float(self.offset_y_spin.value()), float(self.offset_z_spin.value())]
        self.tag_offsets_mm[tag_id] = vals
        self._save_tag_offsets()
        self._refresh_offset_editor(select_id=tag_id)
        self._append_log(f"ID {tag_id} 偏移已立即保存：X={vals[0]:+.1f}, Y={vals[1]:+.1f}, Z={vals[2]:+.1f} mm")

    @pyqtSlot()
    def delete_current_offset_id(self):
        tag_id = self._parse_offset_id()
        if tag_id is None or tag_id not in self.tag_offsets_mm:
            return
        del self.tag_offsets_mm[tag_id]
        self._save_tag_offsets()
        self._refresh_offset_editor()
        self._append_log(f"已删除 ID {tag_id} 的专属偏移；该 ID 现在默认目标点=Tag中心。")


    @pyqtSlot(int)
    def on_profile_changed(self, index):
        if self._loading_profiles or index < 0:
            return
        data = self.profile_combo.itemData(index)
        if not data:
            return
        self._save_option("camera/profile_key", profile_key(data))
        self._append_log(f"分辨率已选择 {self.profile_combo.itemText(index)}，并已保存为下次默认值。")

    @pyqtSlot(object)
    def on_capture_stats(self, payload):
        try:
            self._capture_fps_value = float(payload.get("input_fps", 0.0))
        except Exception:
            self._capture_fps_value = 0.0
        self._update_performance_label()

    @pyqtSlot(object)
    def on_detection_ready(self, payload):
        if self._camera_stopping:
            return
        now = time.monotonic()
        self._latest_detection_captured_at = float(payload.get("captured_at", now))
        self.latest_overlays = list(payload.get("overlays", []))
        self.last_results = list(payload.get("results", []))
        self._latest_detection_received = now
        self._latest_detection_frame_seq = int(payload.get("frame_seq", -1))
        try:
            hz = float(payload.get("detect_hz", 0.0))
            if hz > 0:
                self._detection_fps_value = hz
            self._detection_ms_value = float(payload.get("detect_ms", 0.0))
        except Exception:
            pass

        # Result widgets are intentionally slower than video painting.  They do
        # not need 30 Hz and therefore cannot dominate the GUI event loop.
        if now - self._last_result_ui_update >= 0.10:
            new_ids = {int(r["id"]) for r in self.last_results}
            if not new_ids.issubset(self.detected_tag_ids):
                self.detected_tag_ids.update(new_ids)
                current_id = self._parse_offset_id()
                self._refresh_offset_editor(select_id=current_id)
            self._update_table(self.last_results)
            self._last_result_ui_update = now
        self._update_performance_label()

    @pyqtSlot(str)
    def on_detection_error(self, text):
        self._latest_detection_captured_at = 0.0
        self.results_updated.emit(None)
        self._append_log(text)
        self.statusBar().showMessage(text)

    def _update_performance_label(self):
        seq, captured_at, frame, _k, _d, meta = self.frame_buffer.snapshot()
        if frame is not None:
            src_h, src_w = frame.shape[:2]
        else:
            src_w = int(meta.get("width", 0)) if meta else 0
            src_h = int(meta.get("height", 0)) if meta else 0
        pw, ph = self._last_preview_size
        if captured_at > 0:
            latest_age_ms = max(0.0, (time.monotonic() - captured_at) * 1000.0)
            age_text = f"{latest_age_ms:.0f} ms"
        else:
            age_text = "--"
        source_text = f"{src_w}×{src_h}" if src_w and src_h else "--"
        preview_text = f"{pw}×{ph}" if pw and ph else "--"
        det_hz_text = f"{self._detection_fps_value:.1f}" if self._detection_fps_value > 0 else "--"
        det_ms_text = f"{self._detection_ms_value:.1f}" if self._detection_ms_value > 0 else "--"
        self.performance_label.setText(
            f"相机输入：{self._capture_fps_value:.1f} fps | "
            f"界面显示：{self._preview_fps_value:.1f} fps | "
            f"AprilTag：{det_hz_text} Hz / {det_ms_text} ms\n"
            f"原始：{source_text} | 当前预览：{preview_text} | 最新帧年龄：{age_text} | 只显示最新帧"
        )

    @staticmethod
    def _fit_size(src_w, src_h, dst_w, dst_h):
        src_w, src_h = max(1, int(src_w)), max(1, int(src_h))
        dst_w, dst_h = max(1, int(dst_w)), max(1, int(dst_h))
        # Never enlarge the CPU-side preview above the camera's native pixels.
        # Qt may upscale the finished pixmap for a large/fullscreen monitor.
        scale = min(1.0, dst_w / src_w, dst_h / src_h)
        return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))

    def _draw_preview_overlays(self, image, overlays, sx, sy):
        draw_overlays_inplace(image, overlays, sx, sy)

    def _render_preview_bgr(self, frame, dst_w, dst_h):
        src_h, src_w = frame.shape[:2]
        dst_w, dst_h = max(1, int(dst_w)), max(1, int(dst_h))
        if dst_w == src_w and dst_h == src_h:
            preview = frame.copy()
        else:
            interp = cv2.INTER_AREA if (dst_w < src_w or dst_h < src_h) else cv2.INTER_LINEAR
            preview = cv2.resize(frame, (dst_w, dst_h), interpolation=interp)

        sx = dst_w / float(src_w)
        sy = dst_h / float(src_h)
        now = time.monotonic()
        target_hz = max(1.0, float(self.detect_hz_spin.value()))
        overlay_ttl = max(0.50, 2.5 / target_hz)
        overlays = self.latest_overlays if (now - self._latest_detection_received) <= overlay_ttl else []
        self._draw_preview_overlays(preview, overlays, sx, sy)

        if self._latest_detection_received > 0 and not overlays and (now - self._latest_detection_received) <= overlay_ttl:
            cv2.putText(preview, "No AprilTag detected", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,180,255), 2, cv2.LINE_AA)

        cv2.putText(
            preview,
            f"INPUT {src_w}x{src_h} | PREVIEW {dst_w}x{dst_h} | latest-only",
            (18, max(28, dst_h - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA,
        )
        return preview

    @pyqtSlot()
    def refresh_preview(self):
        seq, captured_at, frame, _k, _d, _meta = self.frame_buffer.snapshot()
        if frame is None or seq == self._last_preview_seq:
            return
        self._last_preview_seq = seq
        self.last_frame = frame
        src_h, src_w = frame.shape[:2]

        fullscreen_active = self.fullscreen_preview is not None and self.fullscreen_preview.isVisible()
        mode = self.preview_mode_combo.currentData() or "fit"

        if fullscreen_active:
            fs_size = self.fullscreen_preview.label.contentsRect().size()
            fw, fh = self._fit_size(src_w, src_h, fs_size.width(), fs_size.height())
            preview_bgr = self._render_preview_bgr(frame, fw, fh)
            rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, fw, fh, rgb.strides[0], QImage.Format.Format_RGB888)
            full_pix = QPixmap.fromImage(qimg)
            self.fullscreen_preview.set_pixmap(full_pix)

            # Reuse the fullscreen pixmap for the embedded preview instead of
            # converting the 1080p source a second time.
            if mode == "1to1":
                self.video_label.setPixmap(full_pix)
                self._last_preview_size = (fw, fh)
            else:
                view_size = self.video_scroll.viewport().size()
                embedded = full_pix.scaled(view_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                self.video_label.setPixmap(embedded)
                self._last_preview_size = (embedded.width(), embedded.height())
        else:
            if mode == "1to1":
                if self.video_label.width() != src_w or self.video_label.height() != src_h:
                    self._apply_preview_mode()
                dw, dh = src_w, src_h
            else:
                view_size = self.video_scroll.viewport().size()
                dw, dh = self._fit_size(src_w, src_h, view_size.width(), view_size.height())

            preview_bgr = self._render_preview_bgr(frame, dw, dh)
            rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, dw, dh, rgb.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix)
            self._last_preview_size = (dw, dh)

        self._preview_fps_frames += 1
        now = time.monotonic()
        elapsed = now - self._preview_fps_t0
        if elapsed >= 1.0:
            self._preview_fps_value = self._preview_fps_frames / elapsed
            self._preview_fps_frames = 0
            self._preview_fps_t0 = now
            self._update_performance_label()

    def _on_result_selection_changed(self):
        row = self.result_table.currentRow()
        if row < 0:
            return
        item = self.result_table.item(row, 0)
        if item is None:
            return
        try:
            self.primary_result_id = int(item.text())
        except Exception:
            return
        self._update_current_target_panel()

    def _update_current_target_panel(self):
        rec = self._latest_display_records.get(self.primary_result_id)
        if rec is None:
            self.current_target_id_label.setText("ID: --")
            self.current_base_x_label.setText("X = -- mm")
            self.current_base_y_label.setText("Y = -- mm")
            self.current_base_z_label.setText("Z = -- mm")
            return
        self.current_target_id_label.setText(f"ID: {self.primary_result_id}")
        bp = rec.get("base_point")
        if bp is None:
            self.current_base_x_label.setText("X = -- mm")
            self.current_base_y_label.setText("Y = -- mm")
            self.current_base_z_label.setText("Z = -- mm")
            return
        x, y, z = (float(v) * 1000.0 for v in bp)
        self.current_base_x_label.setText(f"X = {x:+.1f} mm")
        self.current_base_y_label.setText(f"Y = {y:+.1f} mm")
        self.current_base_z_label.setText(f"Z = {z:+.1f} mm")

    def _update_table(self, results):
        self.result_table.setRowCount(len(results))
        pan, tilt, angle_source = self._angles_for_transform()
        source_text = "文件原值" if self.calibration.active_source == "original" else "编辑副本"
        state = "Base 已停用" if self.calibration.blocked_reason else "Base 可计算"
        self.coord_source_label.setText(
            f"生效：{source_text} | {state} | Legacy：Cmd Pan={pan:.0f}, Tilt={tilt:.0f} | 编码器仅监视")
        self.coord_source_label.setStyleSheet("color:#b00020;" if self.calibration.blocked_reason else "color:#187a2f;")

        self._latest_display_records = {}
        visible_ids = []
        for row, r in enumerate(results):
            tag_id = int(r["id"])
            visible_ids.append(tag_id)
            target_camera = np.array([
                r["target_x_m"], r["target_y_m"], r["target_z_m"]
            ], dtype=float)
            base_point = None
            try:
                base_point = self.calibration.transform_point(target_camera, pan, tilt)
            except Exception as exc:
                self._append_log(f"机械臂基座坐标换算失败：{exc}")

            self._latest_display_records[tag_id] = {
                "base_point": None if base_point is None else np.asarray(base_point, dtype=float),
                "result": r,
            }

            if base_point is None:
                base_xyz = "--"
            else:
                base_xyz = ", ".join(f"{float(v)*1000.0:+.1f}" for v in base_point)

            target_camera_xyz = ", ".join((
                f'{r["target_x_m"]*1000.0:+.1f}',
                f'{r["target_y_m"]*1000.0:+.1f}',
                f'{r["target_z_m"]*1000.0:+.1f}',
            ))
            tag_xyz = ", ".join((
                f'{r["x_m"]*1000.0:+.1f}',
                f'{r["y_m"]*1000.0:+.1f}',
                f'{r["z_m"]*1000.0:+.1f}',
            ))
            values = [str(tag_id), base_xyz, target_camera_xyz, tag_xyz]
            tooltips = [
                f'AprilTag ID = {tag_id}',
                ('目标点机械臂Base坐标：X=' + f'{base_point[0]*1000.0:+.1f} mm, Y={base_point[1]*1000.0:+.1f} mm, Z={base_point[2]*1000.0:+.1f} mm') if base_point is not None else '机械臂Base坐标当前不可用',
                f'目标点相机坐标：X={r["target_x_m"]*1000.0:+.1f} mm, Y={r["target_y_m"]*1000.0:+.1f} mm, Z={r["target_z_m"]*1000.0:+.1f} mm',
                f'相机坐标系中的Tag中心：X={r["x_m"]*1000.0:+.1f} mm, Y={r["y_m"]*1000.0:+.1f} mm, Z={r["z_m"]*1000.0:+.1f} mm',
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(tooltips[col])
                self.result_table.setItem(row, col, item)

        if visible_ids:
            if self.primary_result_id not in visible_ids:
                self.primary_result_id = visible_ids[0]
            for row in range(self.result_table.rowCount()):
                item = self.result_table.item(row, 0)
                if item is not None and int(item.text()) == self.primary_result_id:
                    self.result_table.selectRow(row)
                    break
        else:
            self.primary_result_id = None
            self.result_table.clearSelection()
        self._update_current_target_panel()
        self._update_spatial_view(pan, tilt, angle_source)
        self.results_updated.emit(self.result_snapshot())

    def result_snapshot(self, max_age_s=1.0):
        """GUI-thread API for an embedded full panel; display coordinates in mm.

        The legacy GUI applies CURRENT Cmd controls to the latest detection.
        Use Locator/AprilTagSystem for per-frame Cmd metadata in a custom host.
        """
        captured_at = getattr(self, "_latest_detection_captured_at", 0.0)
        if not captured_at or (max_age_s is not None and time.monotonic() - captured_at > max_age_s):
            return None
        pan, tilt, _ = self._angles_for_transform()
        targets = []
        for tag_id, record in getattr(self, "_latest_display_records", {}).items():
            row, point = record["result"], record["base_point"]
            targets.append(dict(tag_id=tag_id,
                target_camera_mm=[float(row["target_" + a + "_m"]) * 1000 for a in "xyz"],
                target_base_mm=None if point is None else (point * 1000).tolist(),
                tag_camera_mm=[float(row[a + "_m"]) * 1000 for a in "xyz"]))
        return dict(frame_seq=self._latest_detection_frame_seq, captured_at=captured_at,
            pan_cmd_deg=pan, tilt_cmd_deg=tilt, calibration_source=self.calibration.active_source,
            base_unavailable_reason=self.calibration.blocked_reason, targets=targets)

    def _update_spatial_view(self, pan=None, tilt=None, angle_source=None):
        if not hasattr(self, "spatial_view"):
            return
        if pan is None or tilt is None:
            pan, tilt, angle_source = self._angles_for_transform()
        try:
            frames = self.calibration.frame_transforms(float(pan), float(tilt))
            if frames is None:
                self.spatial_view.set_frames(None, self.calibration.blocked_reason or "请加载有效标定数据")
                return
            mode = "文件原值" if self.calibration.active_source == "original" else "编辑副本"
            source = "COMMAND (Legacy-compatible)"
            self.spatial_view.set_frames(
                frames,
                f"{mode} | {source} | Pan={float(pan):.2f}, Tilt={float(tilt):.2f}"
            )
        except Exception as exc:
            self.spatial_view.set_frames(None, f"空间关系计算失败：{exc}")

    @pyqtSlot(str)
    def on_status(self, text):
        self.statusBar().showMessage(text)
        self._append_log(text)

    @pyqtSlot(str)
    def on_camera_info(self, text):
        self.camera_info_label.setText(text)
        self._append_log(text)

    @pyqtSlot(str)
    def on_error(self, text):
        self._latest_detection_captured_at = 0.0
        self.results_updated.emit(None)
        self._append_log(text.replace("\n", " | "))
        self.preview_timer.stop()
        if self.detector_worker and self.detector_worker.isRunning():
            self.detector_worker.stop()
        QMessageBox.critical(self, "相机错误", text)
        if not self.camera_worker:
            self._set_stopped_ui()

    @pyqtSlot()
    def on_camera_worker_finished(self):
        if self.sender() is not self.camera_worker:
            return
        self.camera_worker = None
        self.preview_timer.stop()
        if self.detector_worker and self.detector_worker.isRunning():
            self.detector_worker.stop()
        self._set_stopped_ui()
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    @pyqtSlot()
    def on_detector_worker_finished(self):
        if self.sender() is self.detector_worker:
            self.detector_worker = None
        if self.camera_worker is None:
            self._set_stopped_ui()
            if self._close_pending:
                QTimer.singleShot(0, self.close)

    def _set_stopped_ui(self):
        self._latest_detection_captured_at = 0.0
        self.results_updated.emit(None)
        self.rgb_options_panel.reset()
        idle = not self._camera_running() and not (self.detector_worker and self.detector_worker.isRunning())
        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(False)
        self.profile_combo.setEnabled(idle)
        self.refresh_btn.setEnabled(idle)
        self._update_performance_label()

    def _append_log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{ts}] {text}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render the latest frame at the new preview size on the next timer
        # tick.  Do not perform full-resolution conversion inside resize events.
        self._last_preview_seq = -1

    def closeEvent(self, event):
        if self.camera_worker and self.camera_worker.isRunning():
            self._close_pending = True
            self.stop_camera()
            self.statusBar().showMessage("正在等待相机与参数线程释放设备…")
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.sync()
        try:
            if hasattr(self, "rx_timer"):
                self.rx_timer.stop()
            if hasattr(self, "pantilt_hold_delay_timer"):
                self.pantilt_hold_delay_timer.stop()
            if hasattr(self, "pantilt_hold_repeat_timer"):
                self.pantilt_hold_repeat_timer.stop()
            if hasattr(self, "preview_timer"):
                self.preview_timer.stop()
            if self.fullscreen_preview is not None:
                self.fullscreen_preview.close()
            self.stop_pantilt_hold(save=True, log=False)
            self.pantilt.disconnect()
        except Exception:
            pass
        if self.detector_worker and self.detector_worker.isRunning():
            self.detector_worker.stop()
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
        if self.detector_worker and self.detector_worker.isRunning():
            self._close_pending = True
            event.ignore()
            return
        self.frame_buffer.clear()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
