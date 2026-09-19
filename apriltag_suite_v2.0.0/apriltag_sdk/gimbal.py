"""Optional short-press/hold and Home policy over the unchanged serial protocol."""
from __future__ import annotations
import math
import threading
import time
from .pantilt import PantiltSerial, PantiltError
from .runtime import EventHook


class GimbalController(PantiltSerial):
    def __init__(self):
        super().__init__()
        self.home_angles = (90, 90)
        self.error = EventHook()
        self._jog_stop = threading.Event()
        self._jog_thread = None
        self._has_command = False

    def command(self, pan, tilt):
        result = super().command(pan, tilt)
        self._has_command = True
        return result

    def set_home(self, pan, tilt):
        """Save Home Cmd only. Does not change fitted calibration zeros or move."""
        self.home_angles = self._clamp_target(pan, tilt)

    def home(self):
        self.stop_jog()
        return self.command(*self.home_angles)

    def command_snapshot(self):
        tx = self.tx_snapshot()
        if not self.connected or not self._has_command:
            return dict(pan_cmd_deg=None, tilt_cmd_deg=None)
        return dict(pan_cmd_deg=float(tx["target_pan"]), tilt_cmd_deg=float(tx["target_tilt"]))

    def start_jog(self, pan_direction=0, tilt_direction=0, step=1, speed=20):
        """One step now; after 260ms, move target continuously at speed Cmd units/s."""
        if pan_direction not in (-1, 0, 1) or tilt_direction not in (-1, 0, 1):
            raise ValueError("directions must be -1, 0 or 1")
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in (step, speed)):
            raise ValueError("step and speed must be positive finite values")
        self.stop_jog()
        if not pan_direction and not tilt_direction:
            return
        pan, tilt = self.command(self.pan + pan_direction * step, self.tilt + tilt_direction * step)
        self._jog_stop.clear()

        def run():
            p, t = float(pan), float(tilt)
            if self._jog_stop.wait(0.260):
                return
            last = time.monotonic()
            try:
                while not self._jog_stop.wait(0.05):
                    now = time.monotonic()
                    dt, last = min(0.2, now - last), now
                    p = min(255.0, max(0.0, p + pan_direction * speed * dt))
                    t = min(171.0, max(9.0, t + tilt_direction * speed * dt))
                    self.command(p, t)
            except Exception as exc:
                self.error.emit(str(exc))
        self._jog_thread = threading.Thread(target=run, name="apriltag-jog", daemon=True)
        self._jog_thread.start()

    def stop_jog(self):
        self._jog_stop.set()
        worker = self._jog_thread
        if worker and worker is not threading.current_thread():
            worker.join(1.0)
            if worker.is_alive():
                raise PantiltError("jog worker did not stop")
        self._jog_thread = None

    def disconnect(self):
        self.stop_jog()
        self._has_command = False
        super().disconnect()
