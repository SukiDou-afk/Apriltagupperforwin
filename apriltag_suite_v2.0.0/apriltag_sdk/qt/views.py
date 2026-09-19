from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget, QSizePolicy

class FrameView3D(QWidget):
    """Lightweight interactive 3-D schematic of Base/Pan/Tilt/Camera frames.

    This is a visualization of the calibration chain, not a metrology renderer.
    All transforms are expressed in the robot Base frame and translations are metres.
    Drag with the left mouse button to rotate the view; use the wheel to zoom.
    """

    view_changed = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._frames = None
        self._message = "等待标定链数据"
        self._yaw_deg = -38.0
        self._pitch_deg = 24.0
        self._zoom = 1.0
        self._drag_pos = None

    def set_view(self, yaw_deg: float, pitch_deg: float, zoom: float):
        self._yaw_deg = float(yaw_deg)
        self._pitch_deg = max(-85.0, min(85.0, float(pitch_deg)))
        self._zoom = max(0.25, min(6.0, float(zoom)))
        self.update()

    def view_state(self):
        return self._yaw_deg, self._pitch_deg, self._zoom

    def set_frames(self, frames, message=""):
        self._frames = frames
        self._message = message or ("" if frames else "等待标定链数据")
        self.update()

    @staticmethod
    def _view_rotation(yaw_deg, pitch_deg):
        yaw = np.radians(float(yaw_deg))
        pitch = np.radians(float(pitch_deg))
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return rx @ rz

    def _project_builder(self, points):
        pts = np.asarray(points, dtype=float)
        center = np.mean(pts, axis=0) if len(pts) else np.zeros(3)
        rv = self._view_rotation(self._yaw_deg, self._pitch_deg)
        q = (rv @ (pts - center).T).T if len(pts) else np.zeros((0, 3))
        if len(q):
            span_x = max(float(np.ptp(q[:, 0])), 0.25)
            span_y = max(float(np.ptp(q[:, 1])), 0.25)
        else:
            span_x = span_y = 1.0
        margin = 38.0
        sx = max(1.0, self.width() - 2.0 * margin) / span_x
        sy = max(1.0, self.height() - 2.0 * margin) / span_y
        scale = min(sx, sy) * 0.76 * self._zoom
        screen_center = np.array([self.width() * 0.5, self.height() * 0.53])

        def project(p):
            qp = rv @ (np.asarray(p, dtype=float) - center)
            return QPointF(
                float(screen_center[0] + qp[0] * scale),
                float(screen_center[1] - qp[1] * scale),
            )
        return project

    @staticmethod
    def _draw_arrow(painter, p0, p1, color, width=2):
        pen = QPen(color, width)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        side = 5.0
        back = 10.0
        a = QPointF(p1.x() - back * ux + side * uy, p1.y() - back * uy - side * ux)
        b = QPointF(p1.x() - back * ux - side * uy, p1.y() - back * uy + side * ux)
        painter.drawLine(p1, a)
        painter.drawLine(p1, b)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(248, 249, 250))
        painter.setPen(QPen(QColor(205, 208, 212), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if not self._frames:
            painter.setPen(QColor(95, 99, 104))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message or "等待标定链数据")
            return

        order = [("base", "Base"), ("pan", "Pan"), ("tilt", "Tilt"), ("camera", "Camera")]
        valid = [(key, label, self._frames.get(key)) for key, label in order if self._frames.get(key) is not None]
        if not valid:
            painter.setPen(QColor(95, 99, 104))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message or "标定链不可用")
            return

        origins = [np.asarray(t[:3, 3], dtype=float) for _, _, t in valid]
        max_dist = 0.0
        for a in origins:
            for b in origins:
                max_dist = max(max_dist, float(np.linalg.norm(a - b)))
        axis_len = min(max(max_dist * 0.22, 0.06), 0.30)

        scene_points = list(origins)
        for _, _, t in valid:
            o = np.asarray(t[:3, 3], dtype=float)
            r = np.asarray(t[:3, :3], dtype=float)
            for j in range(3):
                scene_points.append(o + r[:, j] * axis_len)
        project = self._project_builder(scene_points)

        # Draw physical transform chain first.
        painter.setPen(QPen(QColor(115, 115, 115), 1, Qt.PenStyle.DashLine))
        for i in range(len(origins) - 1):
            painter.drawLine(project(origins[i]), project(origins[i + 1]))

        axis_colors = [QColor(210, 55, 55), QColor(42, 145, 73), QColor(45, 92, 190)]
        axis_names = ["X", "Y", "Z"]
        for key, label, t in valid:
            o = np.asarray(t[:3, 3], dtype=float)
            r = np.asarray(t[:3, :3], dtype=float)
            po = project(o)
            painter.setPen(QPen(QColor(30, 30, 30), 1))
            painter.setBrush(QColor(35, 35, 35))
            painter.drawEllipse(po, 3.5, 3.5)
            painter.setFont(QFont("Arial", 9, QFont.Weight.DemiBold))
            painter.drawText(po + QPointF(6, -6), label)
            for j in range(3):
                pe = project(o + r[:, j] * axis_len)
                self._draw_arrow(painter, po, pe, axis_colors[j], 2 if key != "base" else 3)
                painter.setPen(axis_colors[j])
                painter.setFont(QFont("Arial", 8))
                painter.drawText(pe + QPointF(3, -3), axis_names[j])

        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(70, 70, 70))
        legend = "X 红  Y 绿  Z 蓝   |   左键拖动旋转，滚轮缩放   |   虚线：Base→Pan→Tilt→Camera"
        painter.drawText(10, self.height() - 10, legend)
        if self._message:
            painter.drawText(10, 18, self._message)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            delta = pos - self._drag_pos
            self._drag_pos = pos
            self._yaw_deg += float(delta.x()) * 0.45
            self._pitch_deg = max(-85.0, min(85.0, self._pitch_deg + float(delta.y()) * 0.35))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            self.view_changed.emit(self._yaw_deg, self._pitch_deg, self._zoom)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self._zoom = max(0.25, min(6.0, self._zoom * (1.12 ** steps)))
        self.update()
        self.view_changed.emit(self._yaw_deg, self._pitch_deg, self._zoom)
        event.accept()


