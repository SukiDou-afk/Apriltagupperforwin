from __future__ import annotations

import json
import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationError(ValueError):
    pass


def _vector(data: Any, name: str, length: int = 3) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.shape != (length,) or not np.all(np.isfinite(arr)):
        raise CalibrationError(f"{name} 必须是 {length} 个有限数值")
    return arr


def _rotation_axis(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise CalibrationError(f"不支持的旋转轴: {axis}")


def _rotvec_matrix(rotvec: Any) -> np.ndarray:
    r = _vector(rotvec, "rotation_rotvec")
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = r / theta
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def _rpy_matrix(rpy: Any) -> np.ndarray:
    """RPY in radians; convention R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = _vector(rpy, "rotation_rpy")
    return _rotation_axis("z", yaw) @ _rotation_axis("y", pitch) @ _rotation_axis("x", roll)


def matrix_rpy_deg(r):
    """Display/edit representation; R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) > 1e-9:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = 0.0, math.atan2(-r[0, 1], r[1, 1])
    return np.degrees([roll, pitch, yaw])


def _read_rotation(data: dict[str, Any]) -> np.ndarray:
    if "rotation_matrix" in data:
        r = np.asarray(data["rotation_matrix"], dtype=float)
        if r.shape != (3, 3) or not np.all(np.isfinite(r)):
            raise CalibrationError("rotation_matrix 必须是 3x3 有限数值矩阵")
    elif "rotation_rotvec" in data:
        r = _rotvec_matrix(data["rotation_rotvec"])
    elif "rotation_rpy" in data:
        r = _rpy_matrix(data["rotation_rpy"])
    else:
        raise CalibrationError("缺少 rotation_matrix、rotation_rotvec 或 rotation_rpy")
    if abs(np.linalg.det(r) - 1.0) > 0.03 or not np.allclose(r.T @ r, np.eye(3), atol=0.03):
        raise CalibrationError("旋转矩阵不是有效正交矩阵")
    return r


def _transform(rotation: np.ndarray, translation: Any) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = rotation
    t[:3, 3] = _vector(translation, "translation")
    return t


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"无法读取 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("标定文件顶层必须是 JSON 对象")
    return data


@dataclass(frozen=True)
class CameraGimbalCalibration:
    path: str
    t_tilt_camera: np.ndarray
    pan_axis: str
    tilt_axis: str
    pan_sign: float
    tilt_sign: float
    pan_zero_deg: float
    tilt_zero_deg: float
    parent_frame: str
    child_frame: str
    translation_rmse_mm: float | None

    @classmethod
    def load(cls, path: str | Path) -> "CameraGimbalCalibration":
        return cls.from_data(_read_json(path), str(path))

    @classmethod
    def from_data(cls, data, path=""):
        check_kind(data, "camera")
        model = data.get("model")
        if not isinstance(model, dict):
            raise CalibrationError("相机—云台标定缺少 model 字段")
        pan_axis = str(model.get("pan_axis", "z")).lower()
        tilt_axis = str(model.get("tilt_axis", "x")).lower()
        if pan_axis not in ("x", "y", "z") or tilt_axis not in ("x", "y", "z"):
            raise CalibrationError("pan_axis/tilt_axis 必须是 x、y 或 z")
        if pan_axis != "z":
            raise CalibrationError("Legacy 模型的 pan_axis 必须是 z")
        for key in ("pan_sign", "tilt_sign", "pan_zero_deg", "tilt_zero_deg"):
            try:
                value = float(model[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationError(f"model.{key} 缺失或不是数值") from exc
            if not math.isfinite(value):
                raise CalibrationError(f"model.{key} 必须是有限数值")
            if key.endswith("sign") and value not in (-1.0, 1.0):
                raise CalibrationError(f"model.{key} 必须是 +1 或 -1")
        summary = data.get("fit_summary", {})
        rmse = summary.get("translation_rmse_mm") if isinstance(summary, dict) else None
        return cls(
            path=str(Path(path).resolve()),
            t_tilt_camera=_transform(_read_rotation(data), data.get("translation")),
            pan_axis=pan_axis,
            tilt_axis=tilt_axis,
            pan_sign=float(model.get("pan_sign", 1.0)),
            tilt_sign=float(model.get("tilt_sign", 1.0)),
            pan_zero_deg=float(model["pan_zero_deg"]),
            tilt_zero_deg=float(model["tilt_zero_deg"]),
            parent_frame=str(data.get("parent_frame", "tilt_link")),
            child_frame=str(data.get("child_frame", "camera_color_optical_frame")),
            translation_rmse_mm=float(rmse) if rmse is not None else None,
        )

    def panbase_to_camera(self, pan: float, tilt: float) -> np.ndarray:
        """Original calibrated model, preserved exactly for file mode."""
        pan_rad = math.radians(self.pan_sign * (float(pan) - self.pan_zero_deg))
        tilt_rad = math.radians(self.tilt_sign * (float(tilt) - self.tilt_zero_deg))
        t = np.eye(4, dtype=float)
        t[:3, :3] = _rotation_axis(self.pan_axis, pan_rad) @ _rotation_axis(self.tilt_axis, tilt_rad)
        return t @ self.t_tilt_camera


@dataclass(frozen=True)
class ArmGimbalCalibration:
    path: str
    t_arm_panbase: np.ndarray
    parent_frame: str
    child_frame: str
    translation_rmse_mm: float | None

    @classmethod
    def load(cls, path: str | Path) -> "ArmGimbalCalibration":
        return cls.from_data(_read_json(path), str(path))

    @classmethod
    def from_data(cls, data, path=""):
        check_kind(data, "arm")
        parent = str(data.get("parent_frame", ""))
        child = str(data.get("child_frame", ""))
        if parent and parent != "arm_base":
            raise CalibrationError(f"期望 parent_frame=arm_base，实际为 {parent}")
        if child and child != "pan_base":
            raise CalibrationError(f"期望 child_frame=pan_base，实际为 {child}")
        summary = data.get("fit_summary", {})
        rmse = summary.get("translation_rmse_mm") if isinstance(summary, dict) else None
        return cls(
            path=str(Path(path).resolve()),
            t_arm_panbase=_transform(_read_rotation(data), data.get("translation")),
            parent_frame=parent or "arm_base",
            child_frame=child or "pan_base",
            translation_rmse_mm=float(rmse) if rmse is not None else None,
        )


@dataclass
class CalibrationChain:
    camera: CameraGimbalCalibration | None = None
    arm: ArmGimbalCalibration | None = None
    active_source: str = "original"
    blocked_reason: str = ""

    def arm_from_camera(self, pan, tilt):
        if self.blocked_reason or self.camera is None or self.arm is None:
            return None
        return self.arm.t_arm_panbase @ self.camera.panbase_to_camera(pan, tilt)

    def frame_transforms(self, pan, tilt):
        if self.blocked_reason or self.camera is None or self.arm is None:
            return None
        camera = self.camera
        pan_rad = math.radians(camera.pan_sign * (float(pan) - camera.pan_zero_deg))
        tilt_rad = math.radians(camera.tilt_sign * (float(tilt) - camera.tilt_zero_deg))
        rp, rt = np.eye(4), np.eye(4)
        rp[:3, :3] = _rotation_axis(camera.pan_axis, pan_rad)
        rt[:3, :3] = _rotation_axis(camera.tilt_axis, tilt_rad)
        pan_frame = self.arm.t_arm_panbase @ rp
        tilt_frame = pan_frame @ rt
        return {"base": np.eye(4), "pan": pan_frame, "tilt": tilt_frame,
                "camera": tilt_frame @ camera.t_tilt_camera}

    def transform_point(self, point_camera: Any, pan: float, tilt: float) -> np.ndarray | None:
        """Transform one 3-D point from camera optical frame to arm base frame."""
        t = self.arm_from_camera(pan, tilt)
        if t is None:
            return None
        p = _vector(point_camera, "point_camera")
        return t[:3, :3] @ p + t[:3, 3]

    def transform_ridge(self, ridge: dict[str, Any], pan: float, tilt: float) -> dict[str, Any] | None:
        t = self.arm_from_camera(pan, tilt)
        if t is None:
            return None
        r, trans = t[:3, :3], t[:3, 3]

        def point(key: str) -> list[float]:
            return (r @ np.asarray(ridge[key], dtype=float) + trans).tolist()

        direction = r @ np.asarray(ridge["direction"], dtype=float)
        direction /= max(np.linalg.norm(direction), 1e-12)
        return {
            "frame": "arm_base",
            "endpoint_1": point("endpoint_1"),
            "endpoint_2": point("endpoint_2"),
            "midpoint": point("midpoint"),
            "direction": direction.tolist(),
            "transform_arm_from_camera": t.tolist(),
        }


def check_kind(data, expected):
    if not isinstance(data, dict):
        raise CalibrationError("标定内容必须是 JSON 对象")
    is_arm = data.get("parent_frame") == "arm_base" or data.get("child_frame") == "pan_base"
    is_camera = "model" in data or data.get("parent_frame") == "tilt_link"
    if expected == "camera" and is_arm:
        raise CalibrationError("文件类型不匹配：这是云台→机械臂文件，请放入云台→机械臂栏")
    if expected == "arm" and is_camera:
        raise CalibrationError("文件类型不匹配：这是相机→云台文件，请放入相机→云台栏")
    if expected == "camera":
        if data.get("parent_frame", "tilt_link") != "tilt_link":
            raise CalibrationError("相机文件的 parent_frame 必须是 tilt_link")
        if data.get("child_frame", "camera_color_optical_frame") != "camera_color_optical_frame":
            raise CalibrationError("相机文件的 child_frame 必须是 camera_color_optical_frame")


def document_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


class CalibrationDocument:
    """Original immutable-by-convention snapshot plus a separate editable copy."""
    def __init__(self, path, kind):
        self.path = str(Path(path).resolve())
        self.kind = kind
        self.original = _read_json(path)
        self.parse(self.original)
        self.fingerprint = document_hash(self.original)
        self.edited = copy.deepcopy(self.original)

    def parse(self, data):
        cls = CameraGimbalCalibration if self.kind == "camera" else ArmGimbalCalibration
        return cls.from_data(data, self.path)

    def active(self, source):
        return self.original if source == "original" else self.edited

    def reset(self):
        self.edited = copy.deepcopy(self.original)

    def update_pose(self, xyz_mm=None, rpy_deg=None, model=None):
        candidate = copy.deepcopy(self.edited)
        if xyz_mm is not None:
            candidate["translation"] = (_vector(xyz_mm, "translation_mm") / 1000).tolist()
        if rpy_deg is not None:
            radians = np.radians(_vector(rpy_deg, "RPY_deg"))
            candidate["rotation_matrix"] = _rpy_matrix(radians).tolist()
            candidate["rotation_rpy"] = radians.tolist()
            candidate.pop("rotation_rotvec", None)
        if model is not None:
            candidate["model"] = copy.deepcopy(model)
        self.parse(candidate)
        self.edited = candidate


def calibration_readiness(camera_doc, arm_doc):
    """Check presence only; provenance metadata never restricts file selection.

    Each document and its active working copy are numerically validated when
    parsed. source_pantilt_camera and its optional hash are historical metadata.
    """
    missing = []
    if camera_doc is None:
        missing.append("相机→云台")
    if arm_doc is None:
        missing.append("云台→机械臂")
    if missing:
        return "请加载有效的" + "、".join(missing) + "标定数据；Base 输出已停用", True
    return "已加载两份标定数据；按所选数据计算 Base 坐标", False
