import copy
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from calibration import (CalibrationDocument, CalibrationChain, CalibrationError,
                          calibration_readiness, _rpy_matrix, document_hash)
from camera_options import SensorOptions, RGBOptionsWorker, RGBOptionsPanel
from pantilt import PantiltSerial


@pytest.fixture(scope='session')
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    import gc
    from PySide6.QtCore import QCoreApplication, QEvent
    for widget in application.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()
    gc.collect()


@pytest.fixture
def pair(tmp_path):
    camera = dict(parent_frame='tilt_link', child_frame='camera_color_optical_frame',
                  translation=[.05, -.02, .08], rotation_rpy=[.1, -.2, .3],
                  model=dict(pan_axis='z', tilt_axis='x', pan_sign=1, tilt_sign=-1,
                             pan_zero_deg=36.25, tilt_zero_deg=86.09))
    arm = dict(parent_frame='arm_base', child_frame='pan_base',
               translation=[-.0096, -.1666, .00295], rotation_rpy=[0, 0, -1.67],
               source_pantilt_camera='/old/linux/config/pantilt_camera_calibration.json')
    cp, ap = tmp_path/'pantilt_camera_calibration.json', tmp_path/'arm.json'
    cp.write_text(json.dumps(camera)); ap.write_text(json.dumps(arm))
    return CalibrationDocument(cp, 'camera'), CalibrationDocument(ap, 'arm')


def test_legacy_chain_matches_independent_matrix_math(pair):
    cam, arm = pair
    chain = CalibrationChain(cam.parse(cam.original), arm.parse(arm.original))
    for pan, tilt in [(90, 90), (0, 9), (190, 150)]:
        p, q = np.radians([pan - 36.25, -(tilt - 86.09)])
        rz = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
        rx = np.array([[1,0,0],[0,np.cos(q),-np.sin(q)],[0,np.sin(q),np.cos(q)]])
        point = np.array([.1, .2, .5])
        c, a = chain.camera.t_tilt_camera, chain.arm.t_arm_panbase
        expected = a[:3,:3] @ (rz @ rx @ (c[:3,:3] @ point + c[:3,3])) + a[:3,3]
        np.testing.assert_allclose(chain.transform_point(point, pan, tilt), expected, atol=1e-12)
        np.testing.assert_allclose(chain.frame_transforms(pan,tilt)['camera'], chain.arm_from_camera(pan,tilt))


def test_edit_copy_rotation_precedence_and_original_unchanged(pair):
    cam, arm = pair
    before = Path(cam.path).read_bytes()
    original = copy.deepcopy(cam.original)
    cam.update_pose(rpy_deg=[20,30,40], xyz_mm=[100,200,300])
    np.testing.assert_allclose(cam.parse(cam.edited).t_tilt_camera[:3,:3], _rpy_matrix(np.radians([20,30,40])))
    assert Path(cam.path).read_bytes() == before and cam.original == original
    assert cam.active('original') == original
    cam.reset()
    assert cam.edited == original


def test_wrong_file_type_and_bad_values(pair):
    cam, arm = pair
    with pytest.raises(CalibrationError, match='这是云台→机械臂'):
        CalibrationDocument(arm.path, 'camera')
    with pytest.raises(CalibrationError, match='这是相机→云台'):
        CalibrationDocument(cam.path, 'arm')
    for bad in ['', 'xy', 'bad']:
        d=copy.deepcopy(cam.original); d['model']['tilt_axis']=bad
        with pytest.raises(CalibrationError): cam.parse(d)
    d=copy.deepcopy(cam.original);d['model']['pan_zero_deg']=float('nan')
    with pytest.raises(CalibrationError):cam.parse(d)
    d=copy.deepcopy(cam.original);d['rotation_matrix']=[[1,2,3]]*3
    with pytest.raises(CalibrationError):cam.parse(d)


def test_provenance_never_blocks_file_selection(pair):
    cam,arm=pair
    expected = CalibrationChain(cam.parse(cam.original),arm.parse(arm.original)).transform_point([0,0,1],90,90)
    for source in ["different_name.json", "C:/old/path/another.json", None, ""]:
        arm.original['source_pantilt_camera'] = source
        arm.original['source_pantilt_camera_sha256'] = 'different-historical-hash'
        assert not calibration_readiness(cam,arm)[1]
        actual = CalibrationChain(cam.parse(cam.original),arm.parse(arm.original)).transform_point([0,0,1],90,90)
        np.testing.assert_allclose(actual,expected)
    arm.original.pop('source_pantilt_camera')
    assert not calibration_readiness(cam,arm)[1]
    assert calibration_readiness(None,arm)[1]
    assert calibration_readiness(cam,None)[1]


def test_editor_source_switch_persistence_and_failure(app, pair, tmp_path):
    from calibration_editor import CalibrationPanel
    cam,arm=pair
    settings=QSettings(str(tmp_path/'settings.ini'),QSettings.IniFormat)
    panel=CalibrationPanel(settings); chains=[];panel.chain_changed.connect(chains.append)
    panel.editors['camera'].load(cam.path);panel.editors['arm'].load(arm.path)
    original=chains[-1].transform_point([0,0,1],90,90)
    panel.editors['arm'].xyz[0].setValue(100)
    np.testing.assert_allclose(chains[-1].transform_point([0,0,1],90,90),original)
    panel.source.setCurrentIndex(1)
    edited=chains[-1].transform_point([0,0,1],90,90)
    np.testing.assert_allclose(edited-original,[.1096,0,0],atol=1e-9)
    np.testing.assert_allclose(chains[-1].frame_transforms(90,90)['camera'],chains[-1].arm_from_camera(90,90))
    p2=CalibrationPanel(settings);restored=[];p2.chain_changed.connect(restored.append);p2.restore()
    assert p2.source.currentData()=='edited'
    np.testing.assert_allclose(restored[-1].transform_point([0,0,1],90,90),edited)
    panel.editors['camera'].load(arm.path)
    assert chains[-1].transform_point([0,0,1],90,90) is None
    panel.close();p2.close()


def test_file_reload_invalidates_saved_copy(app,pair,tmp_path):
    from calibration_editor import DocumentEditor
    cam,_=pair;s=QSettings(str(tmp_path/'s.ini'),QSettings.IniFormat)
    e=DocumentEditor('camera',s);e.load(cam.path);e.xyz[0].setValue(999)
    changed=copy.deepcopy(cam.original);changed['translation'][0]=.333
    Path(cam.path).write_text(json.dumps(changed))
    e2=DocumentEditor('camera',s);assert e2.load(cam.path,restore=True)
    assert e2.xyz[0].value()==333


class FakeSensor:
    def __init__(self):
        self.values={'enable_auto_exposure':1.,'exposure':20.,'gain':5.,
                     'enable_auto_white_balance':1.,'white_balance':4500.,'sharpness':2.}
        self.writes=[];self.fail=False;self.delay=0
    def supports(self,k):return k in self.values
    def get_option_range(self,k):
        lo,hi,step=(0.,1.,1.) if k.startswith('enable') else ((2800.,6500.,10.) if k=='white_balance' else (0.,100.,1.))
        return SimpleNamespace(min=lo,max=hi,step=step,default=self.values[k])
    def get_option(self,k):return self.values[k]
    def is_option_read_only(self,k):return False
    def get_option_description(self,k):return k
    def set_option(self,k,v):
        time.sleep(self.delay)
        if self.fail:raise RuntimeError('USB error')
        self.values[k]=v;self.writes.append((k,v,threading.get_ident()))


def option_enum():
    from camera_options import OPTION_LABELS
    return SimpleNamespace(**{k:k for k in OPTION_LABELS})


def test_options_order_range_quantization_and_error():
    sensor=FakeSensor();c=SensorOptions(sensor,option_enum())
    assert 'brightness' not in c.snapshot()
    applied,errors=c.apply({'exposure':25.3,'enable_auto_exposure':0})
    assert not errors and applied['exposure']==25
    assert [w[0] for w in sensor.writes]==['enable_auto_exposure','exposure']
    assert c.apply({'gain':float('nan')})[1]
    assert c.apply({'gain':101})[1]
    assert c.apply({'white_balance':5000})[1]
    sensor.fail=True
    assert c.apply({'sharpness':3})[1]
    sensor.fail=False
    assert c.apply({'sharpness':4})[0]['sharpness']==4


def test_option_thread_does_not_block_frame_producer(app):
    from apriltag_upper import LatestFrameBuffer
    sensor=FakeSensor();sensor.delay=.15
    worker=RGBOptionsWorker(sensor,option_enum(),'fake');out=[]
    worker.snapshot_ready.connect(out.append)
    worker.start();worker.request({'enable_auto_exposure':0,'exposure':30})
    buffer=LatestFrameBuffer();start=time.monotonic()
    for i in range(100):buffer.publish(np.full((2,2,3),i,dtype=np.uint8))
    assert time.monotonic()-start<.1
    deadline=time.monotonic()+2
    while time.monotonic()<deadline and not sensor.writes:
        app.processEvents();time.sleep(.01)
    worker.stop();assert worker.wait(2000)
    assert sensor.writes and all(w[2]!=threading.get_ident() for w in sensor.writes)
    assert buffer.snapshot()[0]==100


def test_rgb_panel_restores_by_serial_and_disables_auto_dependents(app,tmp_path):
    s=QSettings(str(tmp_path/'s.ini'),QSettings.IniFormat)
    s.setValue('rgb_options/fake/enable_auto_exposure',0)
    s.setValue('rgb_options/fake/exposure',35)
    panel=RGBOptionsPanel(s);requests=[];panel.requested.connect(requests.append)
    sensor=FakeSensor();controller=SensorOptions(sensor,option_enum())
    payload=dict(serial='fake',records=controller.snapshot(),applied={},errors=[],generation=0)
    panel.update_snapshot(payload)
    assert requests[-1]=={'enable_auto_exposure':0.,'exposure':35.}
    assert not panel.widgets['exposure'].isEnabled()
    applied,errors=controller.apply(requests[-1])
    panel.update_snapshot(dict(payload,records=controller.snapshot(),applied=applied,generation=1))
    assert panel.widgets['exposure'].isEnabled()
    assert float(s.value('rgb_options/fake/exposure'))==35
    panel.reset();requests.clear()
    panel.update_snapshot(dict(payload,serial='other'))
    assert requests==[]


def test_command_frame_and_gbk_every_split():
    assert PantiltSerial.packet(90,50)==bytes.fromhex('FF FE 5A 32 00 00 00 00 00 68')
    raw='舵机A角度：118\r\n舵机B角度：82\r\n'.encode('gbk')
    for split in range(len(raw)+1):
        p=PantiltSerial();p._consume_rx_bytes(raw[:split]);p._consume_rx_bytes(raw[split:])
        assert p.feedback_snapshot()['pan']==118 and p.feedback_snapshot()['tilt']==82
    p=PantiltSerial()
    for byte in raw:p._consume_rx_bytes(bytes([byte]))
    assert p.feedback_snapshot()['pan']==118 and p.feedback_snapshot()['tilt']==82


def test_apriltag_detection_pose_and_per_id_offset():
    import cv2
    from apriltag_upper import DetectionWorker,LatestFrameBuffer,TAG_FAMILY
    image=np.full((700,900),255,np.uint8)
    tag=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_FAMILY),3,220)
    image[200:420,400:620]=tag
    bgr=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    worker=DetectionWorker(LatestFrameBuffer())
    k=np.array([[900.,0,450],[0,900,350],[0,0,1.]])
    _,r0=worker._run_detection(bgr,k,np.zeros(5),50,False,{},960)
    _,r1=worker._run_detection(bgr,k,np.zeros(5),50,False,{3:np.array([100,0,0])},960)
    assert len(r0)==len(r1)==1 and r0[0]['id']==3
    assert .19<r0[0]['z_m']<.22
    a=np.array([r0[0][f'target_{v}_m'] for v in 'xyz']);b=np.array([r1[0][f'target_{v}_m'] for v in 'xyz'])
    assert np.linalg.norm(b-a)==pytest.approx(.1,abs=1e-5)


def test_main_window_cmd_only_and_home_independent(app,tmp_path,monkeypatch):
    import apriltag_upper as gui
    factory=lambda *_:QSettings(str(tmp_path/'gui.ini'),QSettings.IniFormat)
    factory.IniFormat=QSettings.IniFormat
    monkeypatch.setattr(gui,'QSettings',factory)
    monkeypatch.setattr(gui.MainWindow,'refresh_profiles',lambda *a,**k:None)
    w=gui.MainWindow()
    before=w._angles_for_transform()
    w.feedback_pan=118;w.feedback_tilt=82;w.feedback_valid=True
    w.home_pan_spin.setValue(101)
    assert w._angles_for_transform()==before
    assert not hasattr(w,'manual_spins')
    assert not w.calibration.blocked_reason
    w.close()


def test_capture_slow_detection_and_async_close(app,tmp_path,monkeypatch):
    import apriltag_upper as gui
    import pyrealsense2 as rs
    sensor=FakeSensor();sensor.delay=.1
    profile=SimpleNamespace(
        stream_type=lambda:rs.stream.color,
        as_video_stream_profile=lambda:profile,
        get_intrinsics=lambda:SimpleNamespace(fx=500,fy=500,ppx=32,ppy=24,coeffs=[0]*5),
        format=lambda:rs.format.bgr8,width=lambda:64,height=lambda:48,fps=lambda:30)
    sensor.get_stream_profiles=lambda:[profile]
    device=SimpleNamespace(supports=lambda _:True,get_info=lambda _:'fake-camera',query_sensors=lambda:[sensor])
    pipeline_profile=SimpleNamespace(get_device=lambda:device,get_stream=lambda _:profile)
    stopped=[]
    class Pipeline:
        def start(self,_):return pipeline_profile
        def wait_for_frames(self,_):
            time.sleep(.01)
            return SimpleNamespace(get_color_frame=lambda:np.zeros((48,64,3),np.uint8))
        def stop(self):stopped.append(True)
    # A numpy array has ambiguous truthiness, so use an ordinary frame handle.
    Pipeline.wait_for_frames=lambda self,_:(time.sleep(.01) or SimpleNamespace(get_color_frame=lambda:object()))
    monkeypatch.setattr(rs,'pipeline',Pipeline)
    monkeypatch.setattr(gui,'color_frame_to_bgr',lambda *_:np.zeros((48,64,3),np.uint8))
    def slow_detection(*_):time.sleep(.15);return [],[]
    monkeypatch.setattr(gui.DetectionWorker,'_run_detection',slow_detection)
    factory=lambda *_:QSettings(str(tmp_path/'gui.ini'),QSettings.IniFormat)
    factory.IniFormat=QSettings.IniFormat
    monkeypatch.setattr(gui,'QSettings',factory)
    monkeypatch.setattr(gui.MainWindow,'refresh_profiles',lambda *a,**k:None)
    w=gui.MainWindow();w.show()
    w.profile_combo.addItem('fake',(64,48,30,rs.format.bgr8));w.profile_combo.setCurrentIndex(0)
    w.start_camera()
    deadline=time.monotonic()+1
    while time.monotonic()<deadline and w.frame_buffer.snapshot()[0]<15:
        app.processEvents();time.sleep(.01)
    assert w.frame_buffer.snapshot()[0]>=15
    assert w.camera_worker.isRunning()
    w.request_rgb_options({'enable_auto_exposure':0,'exposure':30})
    w.close()
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and w.isVisible():
        app.processEvents();time.sleep(.01)
    assert stopped and not w.isVisible() and w.camera_worker is None


def test_stale_option_snapshot_still_persists_acknowledged_write(app,tmp_path):
    s=QSettings(str(tmp_path/'options.ini'),QSettings.IniFormat)
    panel=RGBOptionsPanel(s);panel.minimum_generation=2
    panel.update_snapshot(dict(serial='fake',records={},applied={'sharpness':4},errors=[],generation=1))
    assert float(s.value('rgb_options/fake/sharpness'))==4
    assert panel.serial==''


def test_user_supplied_files_enable_base_and_view(app,tmp_path):
    from calibration_editor import CalibrationPanel
    s=QSettings(str(tmp_path/'defaults.ini'),QSettings.IniFormat)
    panel=CalibrationPanel(s);chains=[];panel.chain_changed.connect(chains.append)
    panel.restore()
    chain=chains[-1]
    assert not chain.blocked_reason
    assert Path(chain.camera.path).name=='camera_to_gimbal_default.json'
    assert Path(chain.arm.path).name=='cr10a_arm_to_pantilt_charuco_calibration.json'
    point=chain.transform_point([0,0,.5],90,90)
    assert np.all(np.isfinite(point))
    np.testing.assert_allclose(chain.frame_transforms(90,90)['camera'],chain.arm_from_camera(90,90))
    # Historical references, even when deliberately different, cannot disable
    # either original or edited operating source in the real panel.
    doc=panel.editors['arm'].document
    doc.original.pop('source_pantilt_camera',None)
    doc.original['source_pantilt_camera_sha256']='old'
    panel.refresh_chain()
    assert not chains[-1].blocked_reason and panel.export_btn.isEnabled()
    panel.source.setCurrentIndex(1)
    assert not chains[-1].blocked_reason
    panel.editors['arm'].xyz[0].setValue(panel.editors['arm'].xyz[0].value()+10)
    np.testing.assert_allclose(chains[-1].transform_point([0,0,.5],90,90)-point,[.01,0,0],atol=1e-9)
    panel.source.setCurrentIndex(0)
    np.testing.assert_allclose(chains[-1].transform_point([0,0,.5],90,90),point)


def test_saved_camera_example_path_is_respected(app,tmp_path,pair):
    from calibration_editor import CalibrationPanel
    cam,arm=pair
    example=tmp_path/'calibration_examples'/'my_camera.json'
    example.parent.mkdir()
    example.write_text(json.dumps(cam.original))
    s=QSettings(str(tmp_path/'saved.ini'),QSettings.IniFormat)
    s.setValue('calibration/camera_to_gimbal',str(example))
    s.setValue('calibration/gimbal_to_arm',arm.path)
    panel=CalibrationPanel(s);chains=[];panel.chain_changed.connect(chains.append);panel.restore()
    assert chains[-1].camera.path==str(example.resolve())
    assert not chains[-1].blocked_reason
