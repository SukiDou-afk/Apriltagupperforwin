from __future__ import annotations
from collections import defaultdict, deque
import cv2
import numpy as np
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

def build_object_points(tag_size_m: float) -> np.ndarray:
    """Object points for SOLVEPNP_IPPE_SQUARE.

    OpenCV ArUco/AprilTag corners are ordered:
    top-left, top-right, bottom-right, bottom-left.
    """
    h = tag_size_m / 2.0
    return np.array([
        [-h, +h, 0.0],
        [+h, +h, 0.0],
        [+h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float32)


class DetectorEngine:
    """Synchronous AprilTag engine. Caller serializes calls on each instance."""
    def __init__(self):
        self._pose_history = defaultdict(lambda: deque(maxlen=5))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_FAMILY), params)
    def reset(self):
        self._pose_history.clear()

    def detect(
        self,
        frame,
        camera_matrix,
        dist_coeffs,
        tag_size_mm,
        median_enabled,
        target_offsets_mm,
        detect_max_width,
    ):
        if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("image_bgr must be a uint8 HxWx3 array")
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all() or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError("camera_matrix must be a finite 3x3 matrix with positive focal lengths")
        if dist_coeffs.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist_coeffs).all():
            raise ValueError("distortion must contain 4, 5, 8, 12 or 14 finite coefficients")
        if not np.isfinite(tag_size_mm) or tag_size_mm <= 0:
            raise ValueError("tag_size_mm must be positive and finite")
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        native_h, native_w = gray_full.shape[:2]
        scale = min(1.0, float(detect_max_width) / float(max(1, native_w)))
        if scale < 0.999:
            detect_gray = cv2.resize(
                gray_full,
                (max(1, int(round(native_w * scale))), max(1, int(round(native_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detect_gray = gray_full

        corners, ids, _ = self.detector.detectMarkers(detect_gray)
        overlays = []
        results = []
        if ids is None or len(ids) == 0:
            return overlays, results

        if scale < 0.999:
            corners = [np.asarray(c, dtype=np.float32) / scale for c in corners]
        else:
            corners = [np.asarray(c, dtype=np.float32).copy() for c in corners]

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            0.01,
        )
        refined = []
        for c in corners:
            pts = np.asarray(c, dtype=np.float32).reshape(4, 1, 2)
            try:
                cv2.cornerSubPix(gray_full, pts, (5, 5), (-1, -1), criteria)
            except Exception:
                pass
            refined.append(pts.reshape(1, 4, 2))
        corners = refined

        tag_size_m = tag_size_mm / 1000.0
        obj_pts = build_object_points(tag_size_m)
        ids_flat = np.asarray(ids).reshape(-1)

        for marker_corners, marker_id_value in zip(corners, ids_flat):
            marker_id = int(marker_id_value)
            img_pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            t = np.asarray(tvec, dtype=np.float64).reshape(-1)
            if t.size < 3:
                continue
            raw_x, raw_y, raw_z = map(float, t[:3])
            if (not np.isfinite(raw_x) or not np.isfinite(raw_y) or
                    not np.isfinite(raw_z) or raw_z <= 0):
                continue

            if median_enabled:
                self._pose_history[marker_id].append((raw_x, raw_y, raw_z))
                arr = np.asarray(self._pose_history[marker_id], dtype=np.float64).reshape(-1, 3)
                x_m = float(np.median(arr[:, 0]))
                y_m = float(np.median(arr[:, 1]))
                z_m = float(np.median(arr[:, 2]))
            else:
                x_m, y_m, z_m = raw_x, raw_y, raw_z

            distance = float(np.sqrt(x_m*x_m + y_m*y_m + z_m*z_m))
            rot_mat, _ = cv2.Rodrigues(rvec)
            offset_mm = target_offsets_mm.get(marker_id)
            if offset_mm is None:
                offset_mm = np.zeros(3, dtype=np.float64)
            offset_tag_m = np.asarray(offset_mm, dtype=np.float64) / 1000.0
            tag_center_camera_m = np.array([x_m, y_m, z_m], dtype=np.float64)
            target_camera_m = rot_mat @ offset_tag_m + tag_center_camera_m
            target_x_m, target_y_m, target_z_m = map(float, target_camera_m)

            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
            reprojection_error_px = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2)-img_pts)**2, axis=1))))
            result = {
                "rotation_camera_tag": rot_mat.tolist(),
                "rvec_camera_tag": np.asarray(rvec).reshape(3).tolist(),
                "raw_tag_camera_m": [raw_x, raw_y, raw_z],
                "corners_px": img_pts.tolist(),
                "reprojection_error_px": reprojection_error_px,
                "id": marker_id,
                "x_m": x_m, "y_m": y_m, "z_m": z_m,
                "distance_m": distance,
                "target_x_m": target_x_m,
                "target_y_m": target_y_m,
                "target_z_m": target_z_m,
                "offset_x_mm": float(offset_mm[0]),
                "offset_y_mm": float(offset_mm[1]),
                "offset_z_mm": float(offset_mm[2]),
            }

            target_px = None
            try:
                target_img, _ = cv2.projectPoints(
                    target_camera_m.reshape(1, 1, 3).astype(np.float64),
                    np.zeros((3,1), dtype=np.float64),
                    np.zeros((3,1), dtype=np.float64),
                    camera_matrix,
                    dist_coeffs,
                )
                uv = np.asarray(target_img).reshape(-1, 2)[0]
                if np.all(np.isfinite(uv)):
                    target_px = (float(uv[0]), float(uv[1]))
            except Exception:
                pass

            axis_px = None
            try:
                axis_len = tag_size_m * 0.5
                axis_obj = np.array([
                    [0.0, 0.0, 0.0],
                    [axis_len, 0.0, 0.0],
                    [0.0, axis_len, 0.0],
                    [0.0, 0.0, axis_len],
                ], dtype=np.float64)
                axis_img, _ = cv2.projectPoints(
                    axis_obj,
                    np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    camera_matrix,
                    dist_coeffs,
                )
                axis_px = np.asarray(axis_img, dtype=np.float64).reshape(-1, 2).tolist()
            except Exception:
                pass

            overlays.append({
                "id": marker_id,
                "img_pts": img_pts.astype(float).tolist(),
                "target_px": target_px,
                "axis_px": axis_px,
                "result": dict(result),
            })
            results.append(result)

        return overlays, results

