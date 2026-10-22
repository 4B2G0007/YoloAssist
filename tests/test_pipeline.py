"""Synthetic integration test; NOT YOLO detection accuracy evidence."""
import copy,csv,json,tempfile,unittest,wave
from pathlib import Path
from assist.config import DEFAULT

class FakeDetector:
    device='test_fake_not_yolo'
    def __init__(self):self.i=0
    def predict(self,frame):
        self.i+=1
        return [] if self.i%4==0 else [{'class_id':0,'confidence':.9,'xyxy':[120,20,200,220]}]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2,numpy as np
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.video=cls.root/'street1.mp4'
        w=cv2.VideoWriter(str(cls.video),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
        assert w.isOpened()
        for i in range(12):
            image=np.zeros((240,320,3),dtype=np.uint8)
            cv2.rectangle(image,(120,20),(200,220),(100+i*5,150,80),-1)
            w.write(image)
        w.release()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_clip(self,**kwargs):
        from assist.runner import run
        c=copy.deepcopy(DEFAULT);c.update(kwargs.pop('config',{}))
        return run(self.video,c,FakeDetector(),output=self.root/'out',**kwargs)
    def test_complete_and_cross_reference(self):
        r=self.run_clip()
        self.assertEqual((r['status'],r['frames_read'],r['video']['verified_frames']),('completed',12,12))
        folder=Path(r['output_directory'])
        logs=[json.loads(x) for x in (folder/'street1_detections.jsonl').read_text().splitlines()]
        self.assertEqual(len(logs),12);self.assertEqual(logs[3]['objects'],[])
        self.assertAlmostEqual(logs[10]['source_time_s'],1.)
        with (folder/'street1_events.csv').open(encoding='utf-8-sig') as f:events=list(csv.DictReader(f))
        ids={int(e['event_id']) for e in events if e['kind']=='decision'}
        self.assertTrue(ids)
        self.assertTrue(all(d['event_id'] in ids for row in logs for d in row['decisions']))
        self.assertTrue((folder/'review_notes.csv').is_file());self.assertTrue((folder/'run_config.json').is_file())
    def test_stride(self):
        r=self.run_clip(config={'stride':3})
        self.assertEqual((r['frames_inferred'],r['frames_skipped'],r['video']['verified_frames']),(4,8,12))
        logs=[json.loads(x) for x in (Path(r['output_directory'])/'street1_detections.jsonl').read_text().splitlines()]
        self.assertFalse(logs[1]['inferred']);self.assertEqual(logs[1]['objects'],[])
        self.assertEqual(logs[1]['last_inferred_frame'],0)
    def test_limit(self):
        r=self.run_clip(max_frames=3)
        self.assertEqual(r['status'],'limited');self.assertTrue(r['cancel_batch'])
    def test_no_record(self):self.assertIsNone(self.run_clip(record=False)['output_directory'])
    def test_missing_input(self):
        from assist.runner import run
        r=run(self.root/'missing.mp4',copy.deepcopy(DEFAULT),FakeDetector(),output=self.root/'out')
        self.assertEqual(r['status'],'failed')
        self.assertEqual(json.loads((Path(r['output_directory'])/'missing_summary.json').read_text())['status'],'failed')
    def test_voices(self):
        root=Path(__file__).resolve().parents[1]/'audio'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(len(manifest),37)
        for name in manifest:
            with wave.open(str(root/name)) as f:
                self.assertEqual(f.getnchannels(),1);self.assertGreater(f.getnframes()/f.getframerate(),.2)
