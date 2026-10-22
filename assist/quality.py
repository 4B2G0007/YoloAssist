"""Cheap image health + background affine motion gate. No semantic occlusion claims."""
import math
import cv2
import numpy as np

class QualityMonitor:
    def __init__(self, c):
        self.c=c
        self.previous=None
        self.previous_mask=None
        self.previous_time=None
        self.state='ok'
        self.candidate='ok'
        self.candidate_since=None
        self.last_notice=-1e9
        self.reference_sharpness=None

    def check(self, frame, now, candidates=()):
        c=self.c
        if self.previous_time is not None and (now<=self.previous_time or now-self.previous_time>c['motion_max_gap']):
            self.candidate_since=now
        h,w=frame.shape[:2]
        sw=min(320,w);sh=max(32,round(h*sw/w))
        gray=cv2.cvtColor(cv2.resize(frame,(sw,sh)),cv2.COLOR_BGR2GRAY)
        mean=float(gray.mean());std=float(gray.std())
        lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        dark_ratio=float(np.mean(gray<c['quality_dark_pixel']))
        raw='ok'
        if mean<c['quality_dark_mean'] and dark_ratio>c['quality_dark_ratio']:
            raw='dark'
        elif std<c['quality_detail_std'] and lap<c['quality_detail_lap']:
            raw='unclear'
        elif (self.reference_sharpness is not None and self.reference_sharpness>c['quality_blur_reference']
              and lap<c['quality_blur_lap'] and lap/self.reference_sharpness<c['quality_blur_ratio']):
            raw='blurred'
        if raw=='ok':
            self.reference_sharpness=lap if self.reference_sharpness is None else .98*self.reference_sharpness+.02*lap
        mask=np.full_like(gray,255)
        for d in candidates:
            if d['confidence']<c['alert_conf']:continue
            x1,y1,x2,y2=d['xyxy']
            x1=max(0,int(x1/w*sw)-5);x2=min(sw,int(x2/w*sw)+6)
            y1=max(0,int(y1/h*sh)-5);y2=min(sh,int(y2/h*sh)+6)
            mask[y1:y2,x1:x2]=0
        motion={'state':'unknown','reason':'insufficient_history','inliers':0,
                'translation_per_s':None,'rotation_deg_per_s':None,'scale_rate':None}
        if self.previous is not None and self.previous.shape==gray.shape:
            dt=now-self.previous_time
            if 0<dt<=c['motion_max_gap']:
                both=cv2.bitwise_and(mask,self.previous_mask)
                pts=cv2.goodFeaturesToTrack(self.previous,maxCorners=100,qualityLevel=.02,minDistance=7,mask=both)
                if pts is not None and len(pts)>=c['motion_min_features']:
                    nxt,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,pts,None)
                    if nxt is not None and status is not None:
                        good=status.ravel().astype(bool)
                        old=pts[good].reshape(-1,2);new=nxt[good].reshape(-1,2)
                        if len(old)>=c['motion_min_features']:
                            aff,inlier=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2.0)
                            if aff is not None and inlier is not None and int(inlier.sum())>=c['motion_min_features']:
                                n=int(inlier.sum());ratio=n/len(old)
                                if ratio>=c['motion_min_inlier_ratio']:
                                    speed=math.hypot(float(aff[0,2])/sw,float(aff[1,2])/sh)/dt
                                    rotation=abs(math.degrees(math.atan2(float(aff[1,0]),float(aff[0,0]))))/dt
                                    scale=abs(math.hypot(float(aff[0,0]),float(aff[1,0]))-1)/dt
                                    moving=(speed>c['motion_speed'] or rotation>c['motion_rotation'] or scale>c['motion_scale'])
                                    motion.update(state='moving' if moving else 'stable',reason='background_affine',inliers=n,
                                                  translation_per_s=speed,rotation_deg_per_s=rotation,scale_rate=scale)
                                else:motion['reason']='low_inlier_ratio'
                            else:motion['reason']='affine_not_supported'
                        else:motion['reason']='too_few_valid_tracks'
                else:motion['reason']='too_few_background_features'
            else:motion['reason']='history_gap'
        self.previous,self.previous_mask,self.previous_time=gray,mask,now
        if raw!='ok':
            motion.update(state='unknown',reason='image_quality')
        if raw!=self.candidate:
            self.candidate,self.candidate_since=raw,now
        if self.candidate_since is None:self.candidate_since=now
        dwell=c['quality_recover_s'] if raw=='ok' else c['quality_hold_s']
        notice=None;before=self.state
        if raw!=self.state and now-self.candidate_since>=dwell:
            self.state=raw
            notice='quality_recovered' if raw=='ok' else ('quality_dark' if raw=='dark' else 'quality_unclear')
        elif self.state!='ok' and now-self.last_notice>=c['quality_repeat_s']:
            notice='quality_dark' if self.state=='dark' else 'quality_unclear'
        if notice:self.last_notice=now
        return {'state':self.state,'raw_condition':raw,'state_before':before,'notice':notice,
                'usable':self.state=='ok','motion_ok':motion['state']=='stable',
                'motion':motion,'mean':mean,'std':std,'laplacian_variance':lap,'dark_ratio':dark_ratio,
                'candidate_duration_s':now-self.candidate_since}
