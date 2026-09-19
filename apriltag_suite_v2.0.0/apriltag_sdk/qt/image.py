from PyQt6.QtGui import QImage
import cv2


def bgr_to_qimage(image_bgr):
    """Owns its pixels, so the returned QImage outlives the NumPy temporary."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
