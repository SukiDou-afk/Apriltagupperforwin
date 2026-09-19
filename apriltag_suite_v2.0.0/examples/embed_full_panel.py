"""Embed every existing UI feature as a tab in a colleague's PyQt6 application."""
import sys
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QLabel
from apriltag_sdk.qt.app import MainWindow as AprilTagWindow


class RobotHost(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人上位机 · 完整功能页接入示例")
        self.resize(1400, 960)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(QLabel("这里放同事现有的机器人功能"), "机器人")
        self.vision = AprilTagWindow(settings_path="vision_settings.ini")
        self.vision.setWindowFlags(Qt.WindowType.Widget)
        tabs.addTab(self.vision, "视觉与云台")
        tabs.setCurrentWidget(self.vision)

    def closeEvent(self, event):
        # The full panel owns its workers. It ignores close until workers exit.
        if not self.vision.close():
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    host = RobotHost()
    host.show()
    sys.exit(app.exec())
