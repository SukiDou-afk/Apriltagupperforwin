"""Minimal host with external-image simulation or managed D435.

python examples/pyqt6_host.py --simulate
python examples/pyqt6_host.py
Only the host creates QApplication. SDK components can live in your own layouts.
"""
import argparse
import sys
import time
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from apriltag_sdk import AprilTagSystem, annotate
from apriltag_sdk.qt.bridge import QtSystemBridge
from apriltag_sdk.qt.image import bgr_to_qimage
from external_image import synthetic_frame


class Host(QWidget):
    def __init__(self, simulate=False):
        super().__init__()
        self.setWindowTitle("同事上位机接入示例 · PyQt6")
        self.resize(1000, 780)
        self.system = AprilTagSystem()
        self.system.locator.configure(tag_size_mm=50)
        self.system.locator.set_offsets({3: [20, 0, 0]})
        self.bridge = QtSystemBridge(self.system, self)
        self.image = QLabel("点击启动；此示例不发送云台运动指令")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(640, 480)
        self.text = QLabel("等待图像")
        self.text.setWordWrap(True)
        self.button = QPushButton("启动")
        layout = QVBoxLayout(self)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.text)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.start)
        self.bridge.frame_ready.connect(self.show_frame)
        self.bridge.result_ready.connect(self.show_result)
        self.bridge.error.connect(self.text.setText)
        self.bridge.status.connect(self.text.setText)
        self._simulate = simulate
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(100)
        self._sim_timer.timeout.connect(self.simulate)
        self._seq = 0

    def start(self):
        try:
            if self._simulate:
                self._sim_timer.start()
            else:
                self.system.start_camera()
                self.bridge.start()
            self.button.setEnabled(False)
        except Exception as exc:
            self.text.setText(str(exc))

    def simulate(self):
        # Tiny generated image ONLY. Real external-camera recognition belongs in
        # your worker thread, not a GUI timer. See integration guide section 4.
        image, matrix, dist = synthetic_frame()
        self._seq += 1
        batch = self.system.locator.process(image, matrix, dist, frame_seq=self._seq)
        self.show_frame((self._seq, time.monotonic(), annotate(image, batch.overlays), matrix, dist, {}))
        self.show_result(batch)

    def show_frame(self, frame):
        if frame is None:
            self.image.setText("暂无新图像")
            return
        pix = QPixmap.fromImage(bgr_to_qimage(frame[2]))
        self.image.setPixmap(pix.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.FastTransformation))

    def show_result(self, batch):
        if batch is None:
            self.text.setText("结果已过期或相机已停止")
            return
        self.text.setText("\n".join("ID %d | Camera mm %s | Base mm %s" %
            (t.tag_id, tuple(round(v, 2) for v in t.target_camera_mm), t.target_base_mm)
            for t in batch.targets) or "本帧未识别到 Tag")

    def closeEvent(self, event):
        self._sim_timer.stop()
        self.bridge.stop()
        try:
            self.system.close()
        except TimeoutError as exc:
            self.text.setText(str(exc))
            QTimer.singleShot(200, self.close)
            event.ignore()
            return
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="start and close automatically")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = Host(args.simulate)
    window.show()
    if args.smoke_test:
        window.start()
        QTimer.singleShot(500, window.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
