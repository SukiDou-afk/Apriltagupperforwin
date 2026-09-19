"""Public synchronous API. No camera, serial or Qt runtime is imported."""
from __future__ import annotations
import copy
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from .calibration import CalibrationDocument, CalibrationChain, calibration_readiness
from .detection import DetectorEngine


def bundled_calibrations():
    """Example files supplied by the project owner; no automatic device matching."""
    root = Path(__file__).parent / "assets" / "calibration"
    return root / "camera_to_gimbal_default.json", root / "cr10a_arm_to_pantilt_charuco_calibration.json"


@dataclass(frozen=True)
class Target:
    tag_id: int
    tag_camera_mm: Tuple[float, float, float]
    target_camera_mm: Tuple[float, float, float]
    target_base_mm: Optional[Tuple[float, float, float]]
    offset_tag_mm: Tuple[float, float, float]
    rotation_camera_tag: List[List[float]]
    corners_px: List[List[float]]
    reprojection_error_px: float


@dataclass(frozen=True)
class DetectionBatch:
    frame_seq: int
    captured_at: float
    finished_at: float
    detect_ms: float
    pan_cmd_deg: Optional[float]
    tilt_cmd_deg: Optional[float]
    calibration_source: str
    base_unavailable_reason: str
    targets: Tuple[Target, ...]
    overlays: list

    @property
    def age_s(self):
        return max(0.0, time.monotonic() - self.captured_at)

    def to_dict(self):
        """JSON-compatible independent copy; timestamps use time.monotonic()."""
        return asdict(self)


class Locator:
    """Thread-serialized recognition, per-ID offsets, calibration and Base conversion.

    Images are uint8 BGR. K and distortion belong to the supplied image resolution.
    Pan/Tilt are command-space degrees. An absent calibration/Cmd gives Base=None.
    All public distances are mm; low-level CalibrationChain uses metres.
    """
    def __init__(self, tag_size_mm=100.0, median_enabled=True, detect_max_width=960):
        self._lock = threading.RLock()
        self._engine = DetectorEngine()
        self._documents = {"camera": None, "arm": None}
        self._source = "original"
        self._offsets = {}
        self._seq = 0
        self.configure(tag_size_mm, median_enabled, detect_max_width)

    def configure(self, tag_size_mm=None, median_enabled=None, detect_max_width=None):
        with self._lock:
            size = self.tag_size_mm if tag_size_mm is None else float(tag_size_mm)
            width = self.detect_max_width if detect_max_width is None else int(detect_max_width)
            if not np.isfinite(size) or size <= 0 or width < 32:
                raise ValueError("tag_size_mm must be positive; detect_max_width >= 32")
            self.tag_size_mm, self.detect_max_width = size, width
            if median_enabled is not None:
                self.median_enabled = bool(median_enabled)
            self._engine.reset()

    def set_offsets(self, offsets_mm):
        parsed = {}
        for key, value in offsets_mm.items():
            tag_id = int(key)
            arr = np.asarray(value, dtype=float)
            if tag_id < 0 or str(tag_id) != str(key) or arr.shape != (3,) or not np.isfinite(arr).all():
                raise ValueError("offsets must map nonnegative integer IDs to three finite mm values")
            parsed[tag_id] = arr.tolist()
        with self._lock:
            self._offsets = parsed

    def load_calibration(self, kind, path):
        if kind not in ("camera", "arm"):
            raise ValueError("kind must be camera or arm")
        with self._lock:
            # A failed new selection must not silently keep computing from the old one.
            self._documents[kind] = None
            self._documents[kind] = CalibrationDocument(path, kind)

    def set_calibration_source(self, source):
        if source not in ("original", "edited"):
            raise ValueError("source must be original or edited")
        with self._lock:
            self._source = source

    def set_calibration_documents(self, camera_document, arm_document, source="original"):
        """Copy a CalibrationPanel's documents into this locator; UI remains owner."""
        if source not in ("original", "edited"):
            raise ValueError("source must be original or edited")
        documents = {"camera": camera_document, "arm": arm_document}
        for kind, doc in documents.items():
            if doc is not None:
                if doc.kind != kind:
                    raise ValueError("calibration document kind mismatch")
                doc.parse(doc.active(source))
        with self._lock:
            self._documents = copy.deepcopy(documents)
            self._source = source

    def edit_calibration(self, kind, xyz_mm=None, rpy_deg=None, model=None):
        with self._lock:
            doc = self._documents[kind]
            if doc is None:
                raise ValueError("load calibration first")
            doc.update_pose(xyz_mm, rpy_deg, model)

    def reset_calibration_copy(self, kind):
        with self._lock:
            if self._documents[kind] is not None:
                self._documents[kind].reset()

    def calibration_chain(self):
        with self._lock:
            cam, arm = self._documents["camera"], self._documents["arm"]
            reason, blocked = calibration_readiness(cam, arm)
            return CalibrationChain(
                cam.parse(cam.active(self._source)) if cam else None,
                arm.parse(arm.active(self._source)) if arm else None,
                self._source, reason if blocked else "")

    def export_calibrations(self, directory, source="edited"):
        """Create a NEW directory containing selected data; refuse existing paths."""
        if source not in ("original", "edited"):
            raise ValueError("source must be original or edited")
        from .config import save_settings
        with self._lock:
            if any(doc is None for doc in self._documents.values()):
                raise ValueError("load both calibrations first")
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=False)
            for kind, doc in self._documents.items():
                save_settings(folder / (kind + ".json"), doc.active(source))

    def export_state(self):
        with self._lock:
            return copy.deepcopy(dict(tag_size_mm=self.tag_size_mm,
                median_enabled=self.median_enabled, detect_max_width=self.detect_max_width,
                offsets_mm=self._offsets, source=self._source,
                calibrations={k: None if d is None else dict(path=d.path,
                    fingerprint=d.fingerprint, edited=d.edited)
                    for k, d in self._documents.items()}))

    def restore_state(self, state):
        # Fingerprints protect edited-copy restoration ONLY, never pair two files.
        with self._lock:
            self.configure(state.get("tag_size_mm", 100), state.get("median_enabled", True),
                           state.get("detect_max_width", 960))
            self.set_offsets(state.get("offsets_mm", {}))
            for kind in ("camera", "arm"):
                self._documents[kind] = None
                item = state.get("calibrations", {}).get(kind)
                if item:
                    self.load_calibration(kind, item["path"])
                    doc = self._documents[kind]
                    if item.get("fingerprint") == doc.fingerprint and "edited" in item:
                        doc.parse(item["edited"])
                        doc.edited = copy.deepcopy(item["edited"])
            self.set_calibration_source(state.get("source", "original"))

    def process(self, image_bgr, camera_matrix, dist_coeffs, pan_cmd_deg=None,
                tilt_cmd_deg=None, captured_at=None, frame_seq=None):
        started = time.monotonic()
        stamp = started if captured_at is None else float(captured_at)
        if not np.isfinite(stamp):
            raise ValueError("captured_at must be a finite monotonic timestamp")
        with self._lock:
            overlays, rows = self._engine.detect(image_bgr, camera_matrix, dist_coeffs,
                self.tag_size_mm, self.median_enabled, self._offsets, self.detect_max_width)
            chain = self.calibration_chain()
            reason = chain.blocked_reason
            if pan_cmd_deg is None or tilt_cmd_deg is None:
                reason = reason or "缺少 Pan/Tilt Cmd"
            elif not np.isfinite([pan_cmd_deg, tilt_cmd_deg]).all():
                raise ValueError("Pan/Tilt Cmd must be finite degrees")
            targets = []
            for row in rows:
                cam = np.array([row["target_x_m"], row["target_y_m"], row["target_z_m"]])
                base = None if reason else chain.transform_point(cam, pan_cmd_deg, tilt_cmd_deg)
                targets.append(Target(int(row["id"]),
                    tuple(1000 * np.array([row["x_m"], row["y_m"], row["z_m"]])),
                    tuple(1000 * cam), None if base is None else tuple(1000 * base),
                    tuple(row["offset_" + axis + "_mm"] for axis in "xyz"),
                    row["rotation_camera_tag"], row["corners_px"], row["reprojection_error_px"]))
            self._seq += 1
            done = time.monotonic()
            return DetectionBatch(self._seq if frame_seq is None else int(frame_seq), stamp, done,
                (done - started) * 1000, pan_cmd_deg, tilt_cmd_deg, self._source,
                reason, tuple(targets), overlays)
