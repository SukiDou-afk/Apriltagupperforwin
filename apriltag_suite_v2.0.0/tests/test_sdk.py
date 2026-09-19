import copy
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from apriltag_sdk import Locator, AprilTagSystem, bundled_calibrations, annotate, GimbalController
from apriltag_sdk.calibration import CalibrationError
from apriltag_sdk.config import load_settings, save_settings


@pytest.fixture
def scene():
    gray = np.full((700, 900), 255, np.uint8)
    gray[200:420, 400:620] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 3, 220)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), np.array([[900.,0,450],[0,900.,350],[0,0,1.]]), np.zeros(5)


def calibrated_locator():
    loc = Locator(50, median_enabled=False)
    for kind, path in zip(("camera", "arm"), bundled_calibrations()):
        loc.load_calibration(kind, path)
    return loc


def test_core_import_does_not_load_optional_runtimes():
    code = "import sys, apriltag_sdk; assert not any(k.startswith(('PyQt6','PySide','pyrealsense2')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_result_units_offsets_base_and_source(scene):
    loc = calibrated_locator()
    a = loc.process(*scene, pan_cmd_deg=90, tilt_cmd_deg=90).targets[0]
    assert 190 < a.tag_camera_mm[2] < 220
    loc.set_offsets({3: [100, 0, 0]})
    batch = loc.process(*scene, pan_cmd_deg=90, tilt_cmd_deg=90)
    b = batch.targets[0]
    np.testing.assert_allclose(np.array(b.target_camera_mm) - a.tag_camera_mm,
        np.array(b.rotation_camera_tag) @ np.array([100,0,0]), atol=1e-6)
    expected = 1000 * loc.calibration_chain().transform_point(np.array(b.target_camera_mm)/1000,90,90)
    np.testing.assert_allclose(b.target_base_mm, expected)
    loc.edit_calibration("arm", xyz_mm=[100, 200, 300])
    np.testing.assert_allclose(loc.process(*scene,pan_cmd_deg=90,tilt_cmd_deg=90).targets[0].target_base_mm, expected)
    loc.set_calibration_source("edited")
    c = loc.process(*scene,pan_cmd_deg=90,tilt_cmd_deg=90)
    assert c.calibration_source == "edited" and not np.allclose(c.targets[0].target_base_mm,expected)
    assert json.loads(json.dumps(c.to_dict()))["targets"][0]["tag_id"] == 3
    assert c.age_s >= 0 and c.targets[0].reprojection_error_px < 3


def test_no_calibration_or_no_command_and_no_tag_clear_results(scene):
    loc = Locator(50)
    assert loc.process(*scene).targets[0].target_base_mm is None
    loc = calibrated_locator()
    assert loc.process(*scene).targets[0].target_base_mm is None
    blank = np.full_like(scene[0],255)
    assert loc.process(blank,*scene[1:]).targets == ()
    with pytest.raises(ValueError):
        loc.process(*scene, pan_cmd_deg=float("nan"), tilt_cmd_deg=90)


def test_bad_selection_disables_old_base_and_export_does_not_overwrite(scene,tmp_path):
    loc = calibrated_locator()
    original = [p.read_bytes() for p in bundled_calibrations()]
    with pytest.raises(CalibrationError):
        loc.load_calibration("camera", bundled_calibrations()[1])
    assert loc.process(*scene,pan_cmd_deg=90,tilt_cmd_deg=90).targets[0].target_base_mm is None
    loc.load_calibration("camera",bundled_calibrations()[0])
    loc.edit_calibration("camera", xyz_mm=[1,2,3])
    loc.export_calibrations(tmp_path/"export")
    with pytest.raises(FileExistsError): loc.export_calibrations(tmp_path/"export")
    assert [p.read_bytes() for p in bundled_calibrations()] == original
    data = load_settings(tmp_path/"export"/"camera.json")
    np.testing.assert_allclose(data["translation"],[.001,.002,.003])


def test_settings_roundtrip_including_edited_copy_and_missing_file(tmp_path):
    loc = calibrated_locator()
    loc.edit_calibration("camera",xyz_mm=[4,5,6]); loc.set_calibration_source("edited")
    loc.set_offsets({3:[0,100,0]})
    system = AprilTagSystem(locator=loc)
    system.gimbal.set_home(100,80)
    system._rgb_memory = {"my-device": {"sharpness": 2}}
    path=tmp_path/"settings.json";system.save_config(path)
    restored=AprilTagSystem(config_path=path)
    assert restored.locator.export_state()==loc.export_state()
    assert restored.gimbal.home_angles==(100,80)
    assert restored._rgb_memory==system._rgb_memory
    assert not restored.gimbal.connected
    restored.close();system.close()
    bad=load_settings(path);bad["locator"]["calibrations"]["camera"]["path"]=str(tmp_path/"absent.json")
    save_settings(path,bad)
    with pytest.raises(CalibrationError): AprilTagSystem(config_path=path)


def test_edited_copy_fingerprint_does_not_restrict_pair(tmp_path):
    loc=calibrated_locator();state=loc.export_state()
    cam,arm=bundled_calibrations()
    data=load_settings(cam);data["translation"][0]=.123
    replacement=tmp_path/"any_name.json";save_settings(replacement,data)
    state["calibrations"]["camera"]["path"]=str(replacement)
    state["source"]="edited"
    loc.restore_state(state)
    chain=loc.calibration_chain()
    assert not chain.blocked_reason
    assert chain.camera.t_tilt_camera[0,3]==pytest.approx(.123)


def test_preview_does_not_mutate_recognition_input(scene):
    loc=Locator(50);before=scene[0].copy()
    batch=loc.process(*scene)
    drawn=annotate(scene[0],batch.overlays,width=450)
    assert drawn.shape[:2]==(350,450)
    np.testing.assert_array_equal(scene[0],before)


def test_latest_result_age_and_snapshot_ownership(scene):
    system=AprilTagSystem()
    system._latest=system.locator.process(*scene,captured_at=time.monotonic()-2)
    assert system.latest_result() is None
    batch=system.latest_result(max_age_s=None)
    batch.targets[0].rotation_camera_tag[0][0]=123
    assert system.latest_result(max_age_s=None).targets[0].rotation_camera_tag[0][0]!=123
    system.frames.set_stream_info(scene[1],scene[2])
    system.frames.publish(scene[0].copy())
    frame=system.latest_frame();frame[2][:]=0
    assert system.latest_frame()[2].any()
    system.stop_camera();assert system.latest_result() is None and system.latest_frame() is None


def test_managed_workers_latest_frames_restart_and_shutdown(scene,monkeypatch):
    from apriltag_sdk import camera
    from apriltag_sdk.runtime import ThreadWorker,EventHook
    class FakeCapture(ThreadWorker):
        def __init__(self,buffer,*args,**kwargs):
            super().__init__();self.buffer=buffer;self.stop_event=threading.Event()
            self.status=EventHook();self.camera_info=EventHook();self.error=EventHook();self.rgb_options=EventHook()
            self.pose=kwargs["pose_provider"]
        def run(self):
            self.buffer.set_stream_info(scene[1],scene[2])
            while not self.stop_event.wait(.005):self.buffer.publish(scene[0],**self.pose())
        def stop(self):self.stop_event.set()
    monkeypatch.setattr(camera,"CameraCaptureWorker",FakeCapture)
    system=AprilTagSystem(locator=calibrated_locator(),detection_hz=10)
    for _ in range(2):
        system.start_camera(900,700,30)
        deadline=time.monotonic()+2
        while system.latest_result() is None and time.monotonic()<deadline:time.sleep(.01)
        batch=system.latest_result()
        assert batch and batch.targets[0].target_base_mm is None
        assert system.frames.snapshot()[0] >= batch.frame_seq
        system.stop_camera()
        assert not system.running and system.latest_frame() is None


def test_camera_failure_surfaces_and_stops_detector(monkeypatch):
    from apriltag_sdk import camera
    def fail():raise RuntimeError("device removed")
    monkeypatch.setattr(camera.rs,"pipeline",fail)
    system=AprilTagSystem();errors=[];system.error.connect(errors.append)
    system.start_camera(640,480,30)
    deadline=time.monotonic()+2
    while not errors and time.monotonic()<deadline:time.sleep(.01)
    system.stop_camera()
    assert errors and "device removed" in errors[0]
    assert system.latest_result() is None


def test_camera_stop_before_start_does_not_open_device(monkeypatch):
    from apriltag_sdk.camera import CameraCaptureWorker
    from apriltag_sdk.frames import LatestFrameBuffer
    worker=CameraCaptureWorker(LatestFrameBuffer());worker.stop();worker.start()
    assert worker.wait(2000)


def test_jog_home_and_cmd_does_not_use_feedback():
    g=GimbalController()
    g._serial=SimpleNamespace(is_open=True,close=lambda:None)
    assert g.command_snapshot()["pan_cmd_deg"] is None
    g._consume_rx_bytes("舵机A角度：118\r\n舵机B角度：82\r\n".encode("gbk"))
    g.command(90,50)
    assert g.command_snapshot()==dict(pan_cmd_deg=90.,tilt_cmd_deg=50.)
    g.start_jog(1,0,step=2,speed=20);g.stop_jog()
    assert g.pan==92
    g.start_jog(1,0,step=1,speed=40);time.sleep(.38);g.stop_jog()
    assert g.pan>93
    p=g.pan;time.sleep(.08);assert g.pan==p
    g.set_home(100,80);assert g.home()==(100,80)
    g.disconnect();assert g.command_snapshot()["pan_cmd_deg"] is None


def test_rgb_ack_remembered_and_restore_skips_manual_when_auto_enabled():
    system=AprilTagSystem();sent=[]
    system._camera=SimpleNamespace(request_options=lambda v:sent.append(v))
    system._rgb_memory={"one":{"enable_auto_exposure":1,"exposure":30,"sharpness":4}}
    snapshot=dict(serial="one",records={"enable_auto_exposure":{"value":1}},applied={},errors=[],generation=0)
    system._on_rgb_options(snapshot)
    assert sent==[{"enable_auto_exposure":1,"sharpness":4}]
    system._on_rgb_options(dict(snapshot,applied={"sharpness":2},generation=1))
    assert system._rgb_memory["one"]["sharpness"]==2
    assert len(sent)==1


def test_input_validation(scene):
    loc=Locator(50)
    for value in (-1,float("nan"),float("inf")):
        with pytest.raises(ValueError):loc.configure(tag_size_mm=value)
    for value in ({-1:[1,2,3]},{3:[1,2]},{3:[1,2,float("nan")]}):
        with pytest.raises(ValueError):loc.set_offsets(value)
    with pytest.raises(ValueError):loc.set_calibration_source("encoder")
    with pytest.raises(ValueError):loc.process(scene[0].astype(float),*scene[1:])
    with pytest.raises(ValueError):loc.process(scene[0],np.zeros((3,3)),scene[2])
