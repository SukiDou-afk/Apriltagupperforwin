from __future__ import annotations
import cv2
import numpy as np

def draw_overlays_inplace(image, overlays, sx=1.0, sy=1.0):
    if image is None:
        return
    h, w = image.shape[:2]
    for ov in overlays:
        try:
            marker_id = int(ov.get("id", -1))
            r = ov.get("result", {})
            pts = np.asarray(ov.get("img_pts", []), dtype=np.float64).reshape(-1, 2)
            if len(pts) == 4:
                pts_scaled = np.column_stack((pts[:, 0] * sx, pts[:, 1] * sy)).round().astype(np.int32)
                cv2.polylines(image, [pts_scaled.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
                center = np.mean(pts_scaled, axis=0).astype(int)
            else:
                center = np.array([30, 60], dtype=int)

            axis = ov.get("axis_px")
            if axis is not None:
                ap = np.asarray(axis, dtype=np.float64).reshape(-1, 2)
                if len(ap) >= 4:
                    ap[:, 0] *= sx
                    ap[:, 1] *= sy
                    ap = np.rint(ap).astype(int)
                    o = tuple(ap[0])
                    cv2.line(image, o, tuple(ap[1]), (0, 0, 255), 2, cv2.LINE_AA)  # X red
                    cv2.line(image, o, tuple(ap[2]), (0, 255, 0), 2, cv2.LINE_AA)  # Y green
                    cv2.line(image, o, tuple(ap[3]), (255, 0, 0), 2, cv2.LINE_AA)  # Z blue

            target = ov.get("target_px")
            if target is not None:
                tx = int(round(float(target[0]) * sx))
                ty = int(round(float(target[1]) * sy))
                if -10000 < tx < 10000 and -10000 < ty < 10000:
                    cv2.line(image, tuple(center), (tx, ty), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(image, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

            text_x = max(6, min(w - 260, int(center[0]) - 120))
            text_y = max(24, min(h - 58, int(center[1]) - 35))
            cv2.putText(image, f"ID {marker_id}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2, cv2.LINE_AA)
            if r:
                cv2.putText(
                    image,
                    f'TGT {r.get("target_x_m",0.0)*1000:+.0f}, {r.get("target_y_m",0.0)*1000:+.0f}, {r.get("target_z_m",0.0)*1000:+.0f} mm',
                    (text_x, text_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 1, cv2.LINE_AA,
                )
        except Exception:
            continue


def annotate(image_bgr, overlays, width=None):
    """Return a new BGR preview; never draw into the detector's input array."""
    h, w = image_bgr.shape[:2]
    if width is None or width >= w:
        out = image_bgr.copy()
    else:
        if width <= 0:
            raise ValueError("width must be positive")
        out = cv2.resize(image_bgr, (int(width), max(1, round(h*width/w))), interpolation=cv2.INTER_AREA)
    draw_overlays_inplace(out, overlays, out.shape[1]/w, out.shape[0]/h)
    return out
