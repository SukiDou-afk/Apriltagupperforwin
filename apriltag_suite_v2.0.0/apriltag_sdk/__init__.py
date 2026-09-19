"""AprilTag robot SDK. Importing this module does not import Qt or RealSense."""
from .locator import Locator, Target, DetectionBatch, bundled_calibrations
from .system import AprilTagSystem
from .gimbal import GimbalController
from .calibration import CalibrationChain, CalibrationDocument, CalibrationError
from .drawing import annotate

__version__ = "2.0.0"
__all__ = ["Locator", "Target", "DetectionBatch", "AprilTagSystem", "GimbalController",
           "CalibrationChain", "CalibrationDocument", "CalibrationError",
           "bundled_calibrations", "annotate"]
