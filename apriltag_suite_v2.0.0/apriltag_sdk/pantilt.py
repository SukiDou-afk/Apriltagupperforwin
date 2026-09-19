from __future__ import annotations

import codecs
import re
import struct
import threading
import time


class PantiltError(RuntimeError):
    pass


class PantiltSerial:
    """Two-axis pan/tilt serial controller with encoder feedback.

    TX command protocol (10-byte binary frame):
        FF FE pan tilt 00 00 00 00 00 checksum
        checksum = pan ^ tilt

    RX feedback protocol (controller actively pushes GBK/GB2312 text):
        舵机A角度：xxx   -> Pan / horizontal / lower axis
        舵机B角度：yyy   -> Tilt / pitch / upper axis

    v1.7 TX policy:
      * GUI/control code only updates the latest desired target.
      * A dedicated TX thread sends at a bounded maximum rate.
      * Old intermediate targets are overwritten instead of queued.
      * Each target is sent once (the previous 3x + sleep burst is removed).
      * The TX thread never sleeps while holding the serial I/O lock, so the
        background encoder reader can continue receiving feedback.

    Legacy/Base coordinate transforms still use the command-space values.  The
    encoder feedback remains monitoring data only.
    """

    _PAN_RE = re.compile(r"舵机\s*A\s*角度\s*[：:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
    _TILT_RE = re.compile(r"舵机\s*B\s*角度\s*[：:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

    def __init__(self) -> None:
        self._serial = None
        self._io_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._tx_state_lock = threading.Lock()

        self._rx_thread: threading.Thread | None = None
        self._rx_stop = threading.Event()
        self._decoder = codecs.getincrementaldecoder("gbk")(errors="ignore")
        self._rx_text_tail = ""
        self._raw_tail = bytearray()
        self._last_rx_error = ""

        self._tx_thread: threading.Thread | None = None
        self._tx_stop = threading.Event()
        self._tx_event = threading.Event()
        self._tx_rate_hz = 12.0
        self._tx_target_pan = 90
        self._tx_target_tilt = 90
        self._tx_target_seq = 0
        self._tx_sent_seq = 0
        self._last_tx_time: float | None = None
        self._last_tx_pan: int | None = None
        self._last_tx_tilt: int | None = None
        self._last_tx_error = ""

        self.port = ""
        self.baudrate = 115200
        self.pan = 90
        self.tilt = 90

        self._feedback_pan: float | None = None
        self._feedback_tilt: float | None = None
        self._pan_feedback_time: float | None = None
        self._tilt_feedback_time: float | None = None
        self._last_rx_time: float | None = None

    @property
    def connected(self) -> bool:
        ser = self._serial
        return bool(ser is not None and getattr(ser, "is_open", False))

    @staticmethod
    def list_ports() -> list[str]:
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except ImportError:
            return []

    @staticmethod
    def _clamp_target(pan: int, tilt: int) -> tuple[int, int]:
        pan = max(0, min(255, int(round(pan))))
        tilt = max(9, min(171, int(round(tilt))))
        return pan, tilt

    def set_tx_rate_hz(self, hz: float) -> float:
        """Set maximum command transmission rate.  No periodic idle spam occurs."""
        hz = max(1.0, min(30.0, float(hz)))
        with self._tx_state_lock:
            self._tx_rate_hz = hz
        self._tx_event.set()
        return hz

    def _reset_feedback_state(self) -> None:
        with self._feedback_lock:
            self._decoder = codecs.getincrementaldecoder("gbk")(errors="ignore")
            self._rx_text_tail = ""
            self._raw_tail.clear()
            self._last_rx_error = ""
            self._feedback_pan = None
            self._feedback_tilt = None
            self._pan_feedback_time = None
            self._tilt_feedback_time = None
            self._last_rx_time = None

    def _reset_tx_state(self) -> None:
        with self._tx_state_lock:
            self._tx_target_pan = int(self.pan)
            self._tx_target_tilt = int(self.tilt)
            self._tx_target_seq = 0
            self._tx_sent_seq = 0  # connection itself must not move the gimbal
            self._last_tx_time = None
            self._last_tx_pan = None
            self._last_tx_tilt = None
            self._last_tx_error = ""
        self._tx_event.clear()

    def connect(self, port: str, baudrate: int = 115200) -> None:
        try:
            import serial
        except ImportError as exc:
            raise PantiltError("缺少 pyserial，请先安装环境") from exc

        if not str(port).strip():
            raise PantiltError("请选择或输入云台串口，例如 COM3")

        self.disconnect()
        try:
            ser = serial.Serial(
                str(port).strip(),
                int(baudrate),
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,
                write_timeout=0.5,
            )
            try:
                ser.setDTR(True)
                ser.setRTS(True)
            except Exception:
                pass

            self._serial = ser
            self.port = str(port).strip()
            self.baudrate = int(baudrate)
            self._reset_feedback_state()
            self._reset_tx_state()

            time.sleep(0.12)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            self._start_rx_thread()
            self._start_tx_thread()
        except Exception as exc:
            self._serial = None
            raise PantiltError(f"无法打开 {port}: {exc}") from exc

    def _start_rx_thread(self) -> None:
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_loop,
            name="PantiltEncoderRX",
            daemon=True,
        )
        self._rx_thread.start()

    def _start_tx_thread(self) -> None:
        self._tx_stop.clear()
        self._tx_thread = threading.Thread(
            target=self._tx_loop,
            name="PantiltLatestTargetTX",
            daemon=True,
        )
        self._tx_thread.start()

    def disconnect(self) -> None:
        self._rx_stop.set()
        self._tx_stop.set()
        self._tx_event.set()
        rx_thread = self._rx_thread
        tx_thread = self._tx_thread
        self._rx_thread = None
        self._tx_thread = None

        ser = self._serial
        self._serial = None
        if ser is not None:
            try:
                with self._io_lock:
                    ser.close()
            except Exception:
                pass

        current = threading.current_thread()
        for thread in (rx_thread, tx_thread):
            if thread is not None and thread.is_alive() and thread is not current:
                thread.join(timeout=0.4)

    @staticmethod
    def packet(pan: int, tilt: int) -> bytes:
        pan = max(0, min(255, int(pan)))
        tilt = max(0, min(255, int(tilt)))
        return struct.pack(
            "10B",
            0xFF,
            0xFE,
            pan,
            tilt,
            0,
            0,
            0,
            0,
            0,
            pan ^ tilt,
        )

    def command(self, pan: int, tilt: int) -> tuple[int, int]:
        """Update the latest target without blocking the GUI.

        Intermediate targets are intentionally not queued.  The background TX
        thread transmits only the newest target at no more than `_tx_rate_hz`.
        """
        pan, tilt = self._clamp_target(pan, tilt)
        if not self.connected:
            raise PantiltError("云台串口未连接")

        with self._tx_state_lock:
            self.pan = pan
            self.tilt = tilt
            self._tx_target_pan = pan
            self._tx_target_tilt = tilt
            self._tx_target_seq += 1
        self._tx_event.set()
        return pan, tilt

    def _tx_loop(self) -> None:
        while not self._tx_stop.is_set():
            self._tx_event.wait(timeout=0.1)
            self._tx_event.clear()
            if self._tx_stop.is_set():
                break

            while not self._tx_stop.is_set():
                with self._tx_state_lock:
                    target_seq = self._tx_target_seq
                    sent_seq = self._tx_sent_seq
                    rate_hz = max(1.0, float(self._tx_rate_hz))
                    last_tx = self._last_tx_time
                if target_seq <= sent_seq:
                    break

                now = time.monotonic()
                min_interval = 1.0 / rate_hz
                if last_tx is not None:
                    wait_s = min_interval - (now - last_tx)
                    if wait_s > 0:
                        # Wait outside the serial lock.  New targets may arrive
                        # during this period and will replace older ones.
                        self._tx_stop.wait(min(wait_s, 0.2))
                        if self._tx_stop.is_set():
                            break
                        continue

                # Re-read the target immediately before writing so a target that
                # arrived during rate limiting supersedes all older targets.
                with self._tx_state_lock:
                    pan = int(self._tx_target_pan)
                    tilt = int(self._tx_target_tilt)
                    seq = int(self._tx_target_seq)

                ser = self._serial
                if ser is None or not getattr(ser, "is_open", False):
                    break

                payload = self.packet(pan, tilt)
                try:
                    with self._io_lock:
                        written = ser.write(payload)
                    if written is not None and int(written) != len(payload):
                        raise OSError(f"串口仅写入 {written}/{len(payload)} 字节")
                    sent_t = time.monotonic()
                    with self._tx_state_lock:
                        self._tx_sent_seq = max(self._tx_sent_seq, seq)
                        self._last_tx_time = sent_t
                        self._last_tx_pan = pan
                        self._last_tx_tilt = tilt
                        self._last_tx_error = ""
                except Exception as exc:
                    with self._tx_state_lock:
                        self._last_tx_error = str(exc)
                    try:
                        with self._io_lock:
                            ser.close()
                    except Exception:
                        pass
                    break

    def tx_snapshot(self) -> dict:
        now = time.monotonic()
        with self._tx_state_lock:
            target_pan = self._tx_target_pan
            target_tilt = self._tx_target_tilt
            target_seq = self._tx_target_seq
            sent_seq = self._tx_sent_seq
            last_tx = self._last_tx_time
            last_pan = self._last_tx_pan
            last_tilt = self._last_tx_tilt
            rate_hz = self._tx_rate_hz
            err = self._last_tx_error
        return {
            "target_pan": target_pan,
            "target_tilt": target_tilt,
            "last_pan": last_pan,
            "last_tilt": last_tilt,
            "pending": target_seq > sent_seq,
            "tx_rate_hz": rate_hz,
            "last_tx_age_s": None if last_tx is None else max(0.0, now - last_tx),
            "error": err,
        }

    def _consume_rx_bytes(self, data: bytes) -> None:
        if not data:
            return
        now = time.monotonic()
        with self._feedback_lock:
            self._last_rx_time = now
            self._raw_tail.extend(data)
            if len(self._raw_tail) > 512:
                del self._raw_tail[:-512]

            text = self._decoder.decode(data, final=False)
            if text:
                self._rx_text_tail += text
                if len(self._rx_text_tail) > 1024:
                    self._rx_text_tail = self._rx_text_tail[-1024:]

                pan_matches = list(self._PAN_RE.finditer(self._rx_text_tail))
                tilt_matches = list(self._TILT_RE.finditer(self._rx_text_tail))
                if pan_matches:
                    self._feedback_pan = float(pan_matches[-1].group(1))
                    self._pan_feedback_time = now
                if tilt_matches:
                    self._feedback_tilt = float(tilt_matches[-1].group(1))
                    self._tilt_feedback_time = now

                if len(self._rx_text_tail) > 256:
                    self._rx_text_tail = self._rx_text_tail[-256:]

    def _rx_loop(self) -> None:
        while not self._rx_stop.is_set():
            ser = self._serial
            if ser is None or not getattr(ser, "is_open", False):
                break
            try:
                data = b""
                with self._io_lock:
                    waiting = int(getattr(ser, "in_waiting", 0) or 0)
                    if waiting > 0:
                        data = bytes(ser.read(min(waiting, 1024)))
                if data:
                    self._consume_rx_bytes(data)
                else:
                    time.sleep(0.008)
            except Exception as exc:
                with self._feedback_lock:
                    self._last_rx_error = str(exc)
                try:
                    ser.close()
                except Exception:
                    pass
                break

    def feedback_snapshot(self, stale_after_s: float = 1.0) -> dict:
        now = time.monotonic()
        with self._feedback_lock:
            pan = self._feedback_pan
            tilt = self._feedback_tilt
            pan_t = self._pan_feedback_time
            tilt_t = self._tilt_feedback_time
            last_rx_t = self._last_rx_time
            text_tail = self._rx_text_tail
            raw_hex = " ".join(f"{b:02X}" for b in self._raw_tail[-64:])
            err = self._last_rx_error

        pan_age = None if pan_t is None else max(0.0, now - pan_t)
        tilt_age = None if tilt_t is None else max(0.0, now - tilt_t)
        last_rx_age = None if last_rx_t is None else max(0.0, now - last_rx_t)
        both_present = pan is not None and tilt is not None and pan_age is not None and tilt_age is not None
        age_s = max(pan_age, tilt_age) if both_present else None
        online = bool(both_present and age_s <= float(stale_after_s) and self.connected)

        return {
            "pan": pan,
            "tilt": tilt,
            "online": online,
            "age_s": age_s,
            "pan_age_s": pan_age,
            "tilt_age_s": tilt_age,
            "last_rx_age_s": last_rx_age,
            "text_tail": text_tail,
            "raw_hex": raw_hex,
            "error": err,
        }

    def clear_input(self) -> None:
        if not self.connected:
            return
        try:
            with self._io_lock:
                self._serial.reset_input_buffer()
            self._reset_feedback_state()
        except Exception:
            pass
