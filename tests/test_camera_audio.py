"""Hardware-free error-path tests; no claim of camera or speaker verification."""
import copy,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from assist.config import DEFAULT
from assist.sound import AudioPlayer

class CameraAudioTests(unittest.TestCase):
    def test_stale_camera_never_selects_alert(self):
        import numpy as np
        from assist.runner import run
        class Camera:
            def __init__(self,*args):pass
            def read(self,last,timeout):return last+1,np.zeros((240,320,3),dtype=np.uint8),time.monotonic()-5
            def close(self):pass
        class Detector:
            device='fake_stale_test'
            def predict(self,frame):return [{'class_id':0,'confidence':.9,'xyxy':[100,0,200,239]}]
        with patch('assist.runner.LatestCamera',Camera):
            r=run('0',copy.deepcopy(DEFAULT),Detector(),record=False,max_frames=2)
        self.assertEqual(r['prompt_counts'],{})
        self.assertEqual(r['frames_inferred'],2)
    def test_disabled_audio_explicitly_logged(self):
        p=AudioPlayer('audio',enabled=False)
        p.submit({'event_id':7,'time_s':1.2})
        self.assertEqual(p.drain()[0]['playback'],'not_played')
        p.close()
    def test_missing_voice_logs_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p=AudioPlayer(folder,enabled=True)
            p.submit({'event_id':1,'time_s':0,'level':'urgent','direction':'center','class_name':'person'})
            collected=[]
            until=time.monotonic()+2
            while time.monotonic()<until:
                collected.extend(p.drain())
                if any(e['playback']=='failed' for e in collected):break
                time.sleep(.01)
            p.close();collected.extend(p.drain())
            self.assertTrue(any(e['playback']=='failed' for e in collected))
