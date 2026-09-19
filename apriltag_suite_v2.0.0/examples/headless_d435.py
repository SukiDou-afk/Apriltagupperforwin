"""Managed camera, no Qt; Ctrl+C closes camera and serial threads."""
import argparse
import time
from apriltag_sdk import AprilTagSystem, bundled_calibrations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", help="optional D435 device serial")
    parser.add_argument("--port", help="optional pan/tilt serial port, e.g. COM5")
    parser.add_argument("--tag-size-mm", type=float, default=50)
    parser.add_argument("--use-bundled-calibration", action="store_true")
    args = parser.parse_args()
    with AprilTagSystem(config_path="sdk_settings.json") as system:
        system.status.connect(print)
        system.error.connect(print)
        system.locator.configure(tag_size_mm=args.tag_size_mm)
        if args.use_bundled_calibration:
            camera, arm = bundled_calibrations()
            system.locator.load_calibration("camera", camera)
            system.locator.load_calibration("arm", arm)
        if args.port:
            system.gimbal.connect(args.port)
            # No movement is sent on connect. Call command() only in your control flow.
        system.start_camera(serial=args.serial)
        last = None
        try:
            while not system.state_snapshot()["error"]:
                batch = system.latest_result(max_age_s=0.5)
                if batch and batch.frame_seq != last:
                    last = batch.frame_seq
                    for target in batch.targets:
                        print(target.tag_id, "Camera mm:", target.target_camera_mm,
                              "Base mm:", target.target_base_mm)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
