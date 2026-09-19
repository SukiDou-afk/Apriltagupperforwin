"""python examples/external_image.py --synthetic (no devices needed)."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from apriltag_sdk import Locator, annotate


def synthetic_frame():
    gray = np.full((700, 900), 255, np.uint8)
    tag = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 3, 220)
    gray[200:420, 400:620] = tag
    matrix = np.array([[900., 0, 450], [0, 900., 350], [0, 0, 1.]])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), matrix, np.zeros(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--intrinsics", help="JSON: camera_matrix (3x3), dist_coeffs")
    parser.add_argument("--tag-size-mm", type=float, default=50)
    parser.add_argument("--camera-calibration")
    parser.add_argument("--arm-calibration")
    parser.add_argument("--pan-cmd", type=float)
    parser.add_argument("--tilt-cmd", type=float)
    parser.add_argument("--output", default="preview.png")
    args = parser.parse_args()
    if args.synthetic:
        image, matrix, distortion = synthetic_frame()
    else:
        if not args.image or not args.intrinsics:
            parser.error("provide --image and --intrinsics, or --synthetic")
        image = cv2.imread(args.image)
        if image is None:
            parser.error("cannot read image")
        intr = json.loads(Path(args.intrinsics).read_text(encoding="utf-8-sig"))
        matrix, distortion = intr["camera_matrix"], intr["dist_coeffs"]
    locator = Locator(tag_size_mm=args.tag_size_mm, median_enabled=False)
    locator.set_offsets({3: [20, 0, 0]})
    for kind, path in (("camera", args.camera_calibration), ("arm", args.arm_calibration)):
        if path:
            locator.load_calibration(kind, path)
    batch = locator.process(image, matrix, distortion, args.pan_cmd, args.tilt_cmd)
    print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
    if not cv2.imwrite(args.output, annotate(image, batch.overlays)):
        raise RuntimeError("cannot write output image")


if __name__ == "__main__":
    main()
