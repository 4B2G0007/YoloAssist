"""Software-only Raspberry Pi adapter tests; fake camera is NOT hardware evidence."""
import copy
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import main as entry
from assist.config import DEFAULT, load, validate
from assist.sources import LatestCamera, picamera_frame_to_bgr


class ColorTests(unittest.TestCase):
    def test_four_formats_are_three_channel_bgr(self):
        samples = {'XBGR8888':[10,20,30,255], 'XRGB8888':[30,20,10,255],
                   'RGB888':[30,20,10], 'BGR888':[10,20,30]}
        for fmt, values in samples.items():
            with self.subTest(fmt=fmt):
                raw = np.array([[values]],dtype=np.uint8)
                result = picamera_frame_to_bgr(raw,fmt)
                np.testing.assert_array_equal(result,np.array([[[30,20,10]]],np.uint8))
                self.assertEqual(result.shape,(1,1,3))
                self.assertTrue(result.flags.c_contiguous)

    def test_wrong_shape_dtype_or_format_rejected(self):
        for data,fmt in [(None,'XBGR8888'),(np.zeros((2,2,3),np.uint8),'XBGR8888'),
                         (np.zeros((2,2,4),np.float32),'XBGR8888'),
                         (np.zeros((2,2,4),np.uint8),'YUV420')]:
            with self.subTest(fmt=fmt,data_type=type(data)):
                with self.assertRaises(ValueError):picamera_frame_to_bgr(data,fmt)


class LaunchTests(unittest.TestCase):
    def test_pi_no_source_uses_picamera2_cpu_and_team_settings(self):
        c,source,path = entry.configure(entry.parse_args([]),raspberry_pi=True)
        self.assertEqual(source,'picamera2')
        self.assertEqual(Path(path).name,'config.pi.json')
        self.assertEqual(c['device'],'cpu')
        self.assertEqual(c['camera_size'],[640,480])
        self.assertEqual(c['camera_format'],'XBGR8888')

    def test_desktop_still_needs_source(self):
        with self.assertRaisesRegex(ValueError,'--source'):
            entry.configure(entry.parse_args([]),raspberry_pi=False)

    def test_desktop_video_keeps_auto_gpu_selection(self):
        c,s,path = entry.configure(entry.parse_args(['--source','street.mp4']),raspberry_pi=False)
        self.assertEqual((s,c['device']),('street.mp4','auto'))
        self.assertEqual(Path(path).name,'config.v1.1.json')

    def test_explicit_camera_selects_pi_config(self):
        c,s,path = entry.configure(entry.parse_args(['--source','picamera2']),raspberry_pi=False)
        self.assertEqual(c['device'],'cpu')
        self.assertEqual(c['camera_format'],'XBGR8888')

    def test_pi_video_does_not_get_replaced_by_camera(self):
        c,s,_ = entry.configure(entry.parse_args(['--source','street.mp4']),raspberry_pi=True)
        self.assertEqual((s,c['device']),('street.mp4','cpu'))

    def test_pi_rejects_cuda_request(self):
        with self.assertRaisesRegex(ValueError,'CUDA'):
            entry.configure(entry.parse_args(['--device','0']),raspberry_pi=True)

    def test_camera_cli_overrides(self):
        args = entry.parse_args(['--source','picamera2','--camera-index','1',
                                 '--camera-format','RGB888','--camera-fps','15','--autofocus','default'])
        c,_,_ = entry.configure(args,raspberry_pi=True)
        self.assertEqual((c['camera_index'],c['camera_format'],c['camera_fps'],c['camera_autofocus']),
                         (1,'RGB888',15.,'default'))

    def test_file_audio_still_requires_realtime(self):
        with self.assertRaisesRegex(ValueError,'realtime'):
            entry.configure(entry.parse_args(['--source','street.mp4','--audio']),raspberry_pi=False)

    def test_invalid_new_camera_values_rejected(self):
        for key,value in [('camera_index',-1),('camera_index',True),('camera_index',1.5),
                          ('camera_format','YUV420'),('camera_fps',0),('camera_fps',float('nan')),
                          ('camera_fps',121),('camera_autofocus','manual'),('camera_warmup_s',-1)]:
            with self.subTest(key=key,value=value):
                c=copy.deepcopy(DEFAULT);c[key]=value
                with self.assertRaises(ValueError):validate(c)

    def test_rule_thresholds_and_cooldowns_are_unchanged(self):
        old=load(entry.ROOT/'config.v1.1.json');pi=load(entry.ROOT/'config.pi.json')
        changed={k for k in old if old[k]!=pi[k]}
        self.assertEqual(changed,{'device','camera_format'})

    def test_linux_arm_is_not_automatically_raspberry_pi(self):
        with patch.object(entry.sys,'platform','linux'),patch.object(Path,'read_text',return_value='Other ARM board'):
            self.assertFalse(entry.is_raspberry_pi())

    def test_pi_device_tree_is_detected(self):
        with patch.object(entry.sys,'platform','linux'),patch.object(Path,'read_text',return_value='Raspberry Pi 5 Model B Rev 1.0\x00'):
            self.assertTrue(entry.is_raspberry_pi())

    def test_non_linux_does_not_probe_device_tree(self):
        with patch.object(entry.sys,'platform','win32'),patch.object(Path,'read_text') as reader:
            self.assertFalse(entry.is_raspberry_pi());reader.assert_not_called()


class FakePicamera:
    """Bounded fake capture; no real camera and no claim of autofocus quality."""
    def __init__(self,autofocus=True,start_error=False,stop_error=False):
        self.camera_controls={'AfMode':object()} if autofocus else {}
        self.arguments=None;self.applied=[];self.started=False;self.closed=False;self.stopped=False
        self.start_error=start_error;self.stop_error=stop_error
        self.wait=threading.Event()

    def create_video_configuration(self,**kwargs):
        self.arguments=kwargs
        return kwargs

    def configure(self,config):self.config=config
    def set_controls(self,controls):self.applied.append(controls)
    def start(self):
        if self.start_error:raise RuntimeError('simulated camera start failure')
        self.started=True
    def capture_array(self,name):
        self.wait.wait(.005)
        w,h=self.arguments['main']['size']
        fmt=self.arguments['main']['format']
        pixel={'XBGR8888':[10,20,30,255],'XRGB8888':[30,20,10,255],
               'RGB888':[30,20,10],'BGR888':[10,20,30]}[fmt]
        return np.tile(np.array(pixel,np.uint8),(h,w,1))
    def stop(self):
        self.stopped=True
        if self.stop_error:raise RuntimeError('simulated stop failure')
    def close(self):self.closed=True


def fake_modules(fake):
    picam=types.ModuleType('picamera2')
    picam.Picamera2=lambda camera_num=0:fake
    lib=types.ModuleType('libcamera')
    lib.controls=types.SimpleNamespace(AfModeEnum=types.SimpleNamespace(Continuous=2))
    return {'picamera2':picam,'libcamera':lib}


class AdapterTests(unittest.TestCase):
    def options(self,**changes):
        c=load(entry.ROOT/'config.pi.json');c['camera_warmup_s']=0;c.update(changes)
        return c

    def test_camera_settings_color_autofocus_and_cleanup(self):
        fake=FakePicamera()
        with patch.dict(sys.modules,fake_modules(fake)):
            cam=LatestCamera('picamera2',[8,6],options=self.options())
            try:
                serial,frame,_=cam.read(0,2)
                self.assertGreater(serial,0);self.assertEqual(frame.shape,(6,8,3))
                np.testing.assert_array_equal(frame[0,0],[30,20,10])
                self.assertEqual(fake.arguments['main'],{'size':(8,6),'format':'XBGR8888'})
                self.assertEqual(fake.arguments['controls'],{'FrameRate':30.})
                self.assertEqual(fake.applied,[{'AfMode':2}])
                self.assertEqual(fake.arguments['queue'],False)
                self.assertTrue(cam.info['autofocus_applied'])
            finally:cam.close()
        self.assertTrue(fake.stopped and fake.closed)
        self.assertFalse(cam.thread.is_alive())

    def test_non_autofocus_camera_continues(self):
        fake=FakePicamera(autofocus=False)
        with patch.dict(sys.modules,fake_modules(fake)):
            cam=LatestCamera('picamera2',[8,6],options=self.options())
            try:
                cam.read(0,2)
                self.assertFalse(cam.info['autofocus_applied'])
                self.assertTrue(cam.info['warnings']);self.assertEqual(fake.applied,[])
            finally:cam.close()

    def test_default_autofocus_does_not_import_libcamera_controls(self):
        fake=FakePicamera();modules=fake_modules(fake);modules['libcamera']=None
        with patch.dict(sys.modules,modules):
            cam=LatestCamera('picamera2',[8,6],options=self.options(camera_autofocus='default'))
            try:cam.read(0,2);self.assertEqual(fake.applied,[])
            finally:cam.close()

    def test_stop_failure_still_closes_camera(self):
        fake=FakePicamera(stop_error=True)
        with patch.dict(sys.modules,fake_modules(fake)):
            cam=LatestCamera('picamera2',[8,6],options=self.options())
            try:cam.read(0,2)
            finally:cam.close()
        self.assertTrue(fake.stopped and fake.closed)

    def test_start_failure_has_busy_hint_and_cleanup(self):
        fake=FakePicamera(start_error=True)
        with patch.dict(sys.modules,fake_modules(fake)):
            cam=LatestCamera('picamera2',[8,6],options=self.options())
            try:
                with self.assertRaisesRegex(RuntimeError,'Stop stream.py'):cam.read(0,2)
            finally:cam.close()
        self.assertTrue(fake.closed)

    def test_missing_picamera2_has_venv_hint(self):
        with patch.dict(sys.modules,{'picamera2':None}):
            cam=LatestCamera('picamera2',[8,6],options=self.options())
            try:
                with self.assertRaisesRegex(RuntimeError,'system-site-packages'):cam.read(0,2)
            finally:cam.close()

    def test_warmup_can_be_cancelled_and_closed(self):
        fake=FakePicamera()
        with patch.dict(sys.modules,fake_modules(fake)):
            cam=LatestCamera('picamera2',[8,6],options=self.options(camera_warmup_s=10))
            try:
                with self.assertRaises(TimeoutError):cam.read(0,.01)
            finally:cam.close()
            with self.assertRaisesRegex(RuntimeError,'stopped'):cam.read(0,.01)
        self.assertTrue(fake.closed);self.assertFalse(cam.thread.is_alive())

    def test_adapter_flows_through_recording_pipeline(self):
        from assist.runner import run
        class EmptyDetector:
            device='FAKE_NOT_YOLO'
            def predict(self,frame):
                if frame.ndim!=3 or frame.shape[2]!=3:raise ValueError('not BGR')
                return []
        fake=FakePicamera()
        c=self.options();c['camera_size']=[64,48]
        with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,fake_modules(fake)):
            r=run('picamera2',c,EmptyDetector(),output=td,max_frames=3)
            self.assertEqual((r['status'],r['frames_read'],r['video']['verified_frames']),('limited',3,3))
            self.assertEqual(r['camera_configuration']['pixel_format'],'XBGR8888')
            root=Path(r['output_directory'])
            lines=(root/'picamera2_detections.jsonl').read_text().splitlines()
            self.assertEqual(len(lines),3)
            self.assertEqual(json.loads(lines[0])['width'],64)
            self.assertEqual(json.loads((root/'run_config.json').read_text())['configuration']['camera_format'],'XBGR8888')
        self.assertTrue(fake.closed)


if __name__=='__main__':unittest.main()
