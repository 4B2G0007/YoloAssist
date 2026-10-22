import copy,csv,json,tempfile,threading,time,unittest
from pathlib import Path
import cv2
import numpy as np
from assist.config import DEFAULT,CLASSES,NEW_CLASSES,validate
from assist.engine import Engine,polygon_overlap
from assist.quality import QualityMonitor
from assist.interaction import make_summary
from assist.sound import AudioPlayer
from assist.voice_catalog import additions


def box(x1=.1,x2=.3,h=.30,cls=0,y=.25):
    return {'class_id':cls,'confidence':.9,'xyxy':[x1*1000,y*1000,x2*1000,(y+h)*1000]}

def texture():
    rng=np.random.default_rng(2026)
    return cv2.cvtColor(rng.integers(40,220,(240,320),dtype=np.uint8),cv2.COLOR_GRAY2BGR)

class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.c=copy.deepcopy(DEFAULT);self.e=Engine(self.c)
    def step(self,ds,t,**kw):return self.e.update(ds,1000,1000,t,**kw)
    def enter(self,mirror=False,h=.3,motion_ok=True):
        values=[]
        for t,a,b in [(0,.10,.30),(.25,.10,.30),(.50,.18,.38),(.75,.24,.44)]:
            if mirror:a,b=1-b,1-a
            values.append(self.step([box(a,b,h=h)],t,motion_ok=motion_ok))
        return values
    def test_new_classes_and_recording_catalog(self):
        self.assertEqual(len(CLASSES),11);self.assertEqual(len(additions()),37)
        self.assertEqual(CLASSES[58][0],'potted plant')
        self.assertIn('normal_center_potted_plant.wav',additions())
    def test_confirmed_left_entry(self):
        values=self.enter()
        for result in values[:3]:self.assertIsNone(result[2])
        rows,events,ch,_=values[-1]
        self.assertEqual(ch['event_type'],'entry');self.assertEqual(ch['entry_origin'],'left')
        self.assertEqual(ch['clip'],'entry_left_person.wav');self.assertTrue(rows[0]['entry_active'])
    def test_confirmed_right_entry(self):
        self.assertEqual(self.enter(mirror=True)[-1][2]['clip'],'entry_right_person.wav')
    def test_entry_and_near_merged(self):
        rows,events,ch,_=self.enter(h=.5)[-1]
        self.assertEqual(ch['event_type'],'near');self.assertTrue(ch['also_entry'])
        self.assertEqual(len([e for e in events if e['decision']=='selected']),1)
    def test_centre_initially_not_entry(self):
        for t in [0,.25,.5]:
            rows,_,ch,_=self.step([box(.35,.55)],t)
            self.assertFalse(rows[0]['entry_active']);self.assertIsNone(ch)
    def test_single_outside_frame_not_enough(self):
        self.step([box()],0)
        self.step([box(.18,.38)],.25)
        rows,_,ch,_=self.step([box(.24,.44)],.5)
        self.assertFalse(rows[0]['entry_active']);self.assertIsNone(ch)
    def test_motion_unknown_blocks_entry(self):
        self.assertIsNone(self.enter(motion_ok=False)[-1][2])
    def test_motion_break_resets_history(self):
        self.step([box()],0);self.step([box()],.25)
        self.step([box(.18,.38)],.5,motion_ok=False)
        rows,_,ch,_=self.step([box(.24,.44)],.75)
        self.assertFalse(rows[0]['entry_active']);self.assertIsNone(ch)
    def test_missing_breaks_entry_history(self):
        self.step([box()],0);self.step([box()],.25);self.step([],.3)
        self.step([box(.18,.38)],.5)
        self.assertIsNone(self.step([box(.24,.44)],.75)[2])
    def test_polygon_overlap_geometry(self):
        p=[[.25,.25],[.75,.25],[.75,.75],[.25,.75]]
        self.assertAlmostEqual(polygon_overlap([0,0,1,1],p),.25)
        self.assertAlmostEqual(polygon_overlap([.3,.3,.4,.4],p),1)
        self.assertEqual(polygon_overlap([0,0,.1,.1],p),0)
    def test_new_obstacle_near_polygon(self):
        for cls in NEW_CLASSES:
            e=Engine(self.c)
            e.update([box(.4,.6,.5,cls=cls,y=.4)],1000,1000,0)
            rows,_,ch,_=e.update([box(.4,.6,.5,cls=cls,y=.4)],1000,1000,.2)
            self.assertEqual(ch['event_type'],'near');self.assertIn('有',ch['text'])
            self.assertTrue(rows[0]['near_polygon_overlap']>0)
    def test_lateral_or_high_obstacle_not_near(self):
        for candidate in [box(0,.1,.7,cls=56,y=.2),box(.4,.6,.35,cls=56,y=0)]:
            e=Engine(self.c)
            e.update([candidate],1000,1000,0)
            rows,_,ch,_=e.update([candidate],1000,1000,.2)
            self.assertFalse(rows[0]['in_roi']);self.assertIsNone(ch)
    def test_walk_vs_classic(self):
        self.assertIsNone(self.step([box()],0)[2])
        self.e.set_mode('classic')
        self.assertIsNotNone(self.step([box()],.2)[2])
    def test_persistent_near_repeats_only_after_long_interval(self):
        candidate=box(.4,.6,.5)
        self.step([candidate],0)
        selected=[]
        for i in range(1,131):
            ch=self.step([candidate],i/10)[2]
            if ch:selected.append(ch['time_s'])
        self.assertEqual(len(selected),2);self.assertGreaterEqual(selected[1]-selected[0],12)
    def test_scene_survives_track_id_change(self):
        candidate=box(.4,.6,.5)
        self.step([candidate],0);self.step([candidate],.1)
        self.step([candidate],1.7)
        rows,ev,ch,_=self.step([candidate],1.8)
        self.assertNotEqual(rows[0]['track_id'],1);self.assertIsNone(ch)
        self.assertEqual(ev[0]['reason'],'scene_already_announced')
    def test_spatially_distinct_new_target_not_scene_suppressed(self):
        self.c['urgent_cooldown']=.01;self.c['urgent_global_gap']=.01;self.e=Engine(self.c)
        self.step([box(.31,.41,.75,y=.1)],0)
        _,_,ch,_=self.step([box(.31,.41,.75,y=.1),box(.59,.69,.75,y=.1)],.2)
        self.assertIsNotNone(ch);self.assertEqual(ch['track_id'],2)
    def test_perception_gate_does_not_consume_alert(self):
        c=box(.4,.6,.75,y=.1)
        _,ev,ch,_=self.step([c],0,perception_ok=False)
        self.assertIsNone(ch);self.assertEqual(ev[0]['reason'],'image_quality_degraded')
        self.assertIsNotNone(self.step([c],.2)[2])
    def test_summary_limit_and_no_counts(self):
        self.e.set_mode('classic')
        rows,_,_,_=self.step([box(.1,.2,cls=0),box(.4,.6,cls=2),box(.75,.9,cls=3),box(.75,.9,cls=56)],0)
        event=make_summary(self.e,rows,1,True)
        self.assertEqual(len(event['clips']),3);self.assertEqual(event['priority'],0)
        self.assertNotIn('安全',event['text'])
    def test_summary_invalid_and_cooldown(self):
        e=make_summary(self.e,[],0,False)
        self.assertEqual(e['clip'],'summary_unavailable.wav')
        self.assertEqual(make_summary(self.e,[],1,True)['decision'],'suppressed')
        self.assertEqual(make_summary(self.e,[],4,True)['clip'],'summary_empty.wav')
    def test_bad_polygon_and_mode_rejected(self):
        for key,val in [('near_polygon',[[0,0],[1,1],[0,1],[1,0]]),('alert_mode','invented')]:
            c=copy.deepcopy(self.c);c[key]=val
            with self.assertRaises(ValueError):validate(c)

class QualityTests(unittest.TestCase):
    def setUp(self):self.q=QualityMonitor(copy.deepcopy(DEFAULT))
    def test_texture_stable_motion(self):
        image=texture();self.q.check(image,0)
        q=self.q.check(image,.2)
        self.assertEqual(q['motion']['state'],'stable');self.assertTrue(q['motion_ok'])
    def test_fast_pan_rejected(self):
        image=texture();self.q.check(image,0)
        q=self.q.check(np.roll(image,20,axis=1),.1)
        self.assertEqual(q['motion']['state'],'moving');self.assertFalse(q['motion_ok'])
    def test_sparse_samples_do_not_prove_continuous_darkness(self):
        dark=np.zeros((240,320,3),np.uint8)
        self.q.check(dark,0)
        q=self.q.check(dark,5)
        self.assertEqual(q['state'],'ok');self.assertIsNone(q['notice'])
    def test_dark_debounce_repeat_recovery(self):
        dark=np.zeros((240,320,3),np.uint8)
        self.assertIsNone(self.q.check(dark,0)['notice'])
        self.assertIsNone(self.q.check(dark,.9)['notice'])
        self.assertEqual(self.q.check(dark,1.01)['notice'],'quality_dark')
        self.assertIsNone(self.q.check(dark,5)['notice'])
        self.assertEqual(self.q.check(dark,22)['notice'],'quality_dark')
        self.q.check(texture(),23)
        self.assertEqual(self.q.check(texture(),23.8)['notice'],'quality_recovered')
    def test_white_wall_is_not_claimed_occlusion(self):
        image=np.full((240,320,3),200,np.uint8)
        self.q.check(image,0);q=self.q.check(image,1.1)
        self.assertEqual(q['state'],'unclear');self.assertEqual(q['motion']['state'],'unknown')
        self.assertNotIn('遮',additions()[q['notice']+'.wav'])
    def test_foreground_mask_insufficient_features(self):
        image=texture();ds=[{'confidence':.9,'xyxy':[0,0,320,240]}]
        self.q.check(image,0,ds);q=self.q.check(image,.2,ds)
        self.assertEqual(q['motion']['state'],'unknown')
    def test_blur_against_texture_reference(self):
        yy,xx=np.indices((240,320))
        gray=(((xx//12+yy//12)%2)*180+40).astype(np.uint8)
        image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        self.q.check(image,0)
        blurred=cv2.GaussianBlur(image,(21,21),6)
        self.assertEqual(self.q.check(blurred,.2)['raw_condition'],'blurred')

class NewAudioTests(unittest.TestCase):
    def test_missing_second_clip_never_starts_partial_summary(self):
        root=Path(__file__).resolve().parents[1]/'audio'
        p=AudioPlayer(root,True)
        p.submit({'event_id':1,'time_s':0,'event_type':'summary','priority':0,
                  'clips':['normal_center_person.wav','not_recorded.wav']})
        events=p.drain();p.close()
        self.assertTrue(any(e['playback']=='failed' and e['reason']=='missing_voice' for e in events))
        self.assertFalse(any(e['playback']=='started' for e in events))
    def test_summary_yields_between_clips(self):
        root=Path(__file__).resolve().parents[1]/'audio'
        player=AudioPlayer(root,False,max_age=10)
        player.enabled=True
        began=threading.Event();release=threading.Event()
        calls=[]
        def fake_play(path):
            calls.append(path.name)
            if len(calls)==1:began.set();release.wait(2)
        player._play=fake_play
        player.thread=threading.Thread(target=player._work,daemon=True);player.thread.start()
        player.submit({'event_id':1,'time_s':0,'event_type':'summary','priority':0,
                       'clips':['normal_center_person.wav','normal_right_car.wav']})
        self.assertTrue(began.wait(1))
        player.submit({'event_id':2,'time_s':.1,'level':'urgent','priority':3,
                       'clips':['urgent_center_person.wav']})
        release.set();events=[];until=time.monotonic()+2
        while time.monotonic()<until:
            events.extend(player.drain())
            if any(e['event_id']==2 and e['playback']=='finished' for e in events):break
            time.sleep(.01)
        player.close();events.extend(player.drain())
        self.assertEqual(calls,['normal_center_person.wav','urgent_center_person.wav'])
        self.assertTrue(any(e['event_id']==1 and e['playback']=='dropped' for e in events))

class NewPipelineTests(unittest.TestCase):
    def test_health_summary_mode_and_event_ids(self):
        from assist.runner import run
        class Empty:
            device='fake_no_objects'
            def predict(self,frame):return []
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'quality.mp4'
            w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
            self.assertTrue(w.isOpened())
            for i in range(30):w.write(np.zeros((240,320,3),np.uint8) if i<13 else texture())
            w.release()
            # This case tests missing audio explicitly; the shipped pack now has all 74 clips.
            r=run(source,copy.deepcopy(DEFAULT),Empty(),output=Path(td)/'out',summary_at=[1.1],
                  audio_dir=Path(td)/'intentionally_missing_audio',
                  key_provider=lambda n:ord('m') if n==15 else -1)
            self.assertEqual(r['status'],'completed');self.assertEqual(r['final_alert_mode'],'classic')
            self.assertEqual(r['video']['verified_frames'],30)
            root=Path(r['output_directory'])
            with (root/'quality_events.csv').open(encoding='utf-8-sig') as f:events=list(csv.DictReader(f))
            decisions=[e for e in events if e['kind']=='decision']
            self.assertTrue(any(e['event_type']=='quality_dark' for e in decisions))
            self.assertTrue(any(e['event_type']=='quality_recovered' for e in decisions))
            summary=next(e for e in decisions if e['event_type']=='summary')
            self.assertEqual(summary['clip'],'summary_unavailable.wav')
            self.assertEqual(len({e['event_id'] for e in decisions}),len(decisions))
            self.assertEqual(summary['voice_available'],'False')

class StaleSummaryTests(unittest.TestCase):
    def test_skipped_frame_does_not_make_stale_result_valid(self):
        from unittest.mock import patch
        from assist.runner import run
        class Camera:
            def __init__(self,*args):pass
            def read(self,last,timeout):return last+1,texture(),time.monotonic()-5
            def close(self):pass
        class Detector:
            device='fake_stale_summary'
            def predict(self,frame):return []
        c=copy.deepcopy(DEFAULT);c['stride']=2
        with tempfile.TemporaryDirectory() as td,patch('assist.runner.LatestCamera',Camera):
            r=run('0',c,Detector(),output=td,max_frames=2,key_provider=lambda n:ord('s') if n==2 else -1)
            folder=Path(r['output_directory'])
            with (folder/'camera0_events.csv').open(encoding='utf-8-sig') as f:events=list(csv.DictReader(f))
            summary=next(e for e in events if e['event_type']=='summary')
            self.assertEqual(summary['clip'],'summary_unavailable.wav')
