"""Reproducible rule demos with SCRIPTED detections; never an accuracy benchmark."""
import argparse,copy,json,sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assist.config import DEFAULT
from assist.runner import run
from assist.recording import dump

class ScriptedDetector:
    device='SCRIPTED_TEST_NOT_YOLO'
    def __init__(self,case):self.case=case;self.index=0
    def predict(self,frame):
        i=self.index;self.index+=1
        if self.case=='quality':return []
        if self.case in ['entry','pan']:
            a=.10 if i<3 else min(.35,.10+.04*(i-2));b=a+.20;y=.25;h=.30;cls=0
        elif self.case=='chair':a=.40;b=.60;y=.4;h=.5;cls=56
        else:a=.40;b=.60;y=.25;h=.5;cls=0
        height,width=frame.shape[:2]
        return [{'class_id':cls,'confidence':.95,'xyxy':[a*width,y*height,b*width,(y+h)*height]}]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='verification-v1.1/scenarios')
    args=parser.parse_args();root=Path(args.output);inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026)
    gray=rng.integers(40,220,(240,320),dtype=np.uint8)
    textured=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    reports=[]
    for case,count in [('entry',20),('pan',20),('chair',20),('quality',20),('persistent',75)]:
        path=inputs/(case+'.mp4')
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(320,240))
        if not w.isOpened():raise RuntimeError('Cannot write fixture')
        for i in range(count):
            frame=textured.copy()
            if case=='quality' and i<=5:frame[:]=0
            if case=='pan':frame=np.roll(frame,i*25,axis=1)
            cv2.putText(frame,'SCRIPTED TEST - NOT YOLO',(5,230),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            w.write(frame)
        w.release()
        modes=['classic','walk'] if case=='persistent' else ['walk']
        for mode in modes:
            c=copy.deepcopy(DEFAULT);c['alert_mode']=mode
            r=run(path,c,ScriptedDetector(case),output=root/mode,audio_dir=Path(__file__).resolve().parents[1]/'audio',summary_at=[1.5])
            reports.append(r)
    def only(case,mode):return next(r for r in reports if Path(r['source']).stem==case and r['final_alert_mode']==mode)
    assert all(r['status']=='completed' for r in reports)
    assert only('entry','walk')['event_counts'].get('entry_selected')==1
    assert only('pan','walk')['event_counts'].get('entry_selected',0)==0
    assert only('chair','walk')['event_counts'].get('near_selected')==1
    assert only('quality','walk')['event_counts'].get('quality_dark_selected')==1
    assert only('quality','walk')['event_counts'].get('quality_recovered_selected')==1
    classic=only('persistent','classic')['event_counts'].get('near_selected',0)
    walk=only('persistent','walk')['event_counts'].get('near_selected',0)
    assert walk<classic
    dump(root/'scenario_assertions.json',{'detector':'SCRIPTED_TEST_NOT_YOLO','all_assertions_passed':True,
        'persistent_near_selected_classic':classic,'persistent_near_selected_walk':walk,'reports':reports})
    print('Scenario assertions passed. Persistent near reminders:',classic,'classic vs',walk,'walk.')
if __name__=='__main__':main()
