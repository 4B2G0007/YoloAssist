import os
from pathlib import Path

class Overlay:
    def __init__(self, c):
        import cv2
        self.cv2 = cv2
        self.c = c
        self.font = None
        fonts = [c['font'], os.environ.get('WINDIR', 'C:/Windows')+'/Fonts/msjh.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                 '/System/Library/Fonts/PingFang.ttc']
        for path in fonts:
            if path and Path(path).is_file():
                try:
                    from PIL import ImageFont
                    self.font = ImageFont.truetype(path, c['font_size'])
                    break
                except OSError:
                    pass

    def draw(self, frame, rows, title, prompt, age=None, quality=None):
        cv2 = self.cv2
        out = frame.copy()
        h,w = out.shape[:2]
        c = self.c
        r = c['roi']
        cv2.rectangle(out, (int(r[0]*w),int(r[1]*h)),(int(r[2]*w)-1,int(r[3]*h)-1),(255,200,0),2)
        import numpy as np
        polygon=np.array([[round(x*w),round(y*h)] for x,y in c['near_polygon']],dtype=np.int32)
        cv2.polylines(out,[polygon],True,(200,100,255),2)
        for x in [w//3,2*w//3]:
            cv2.line(out,(x,0),(x,h),(100,100,100),1)
        for d in rows:
            if not d['eligible']:
                continue
            color = (20,20,240) if d['state']=='near' else ((0,180,255) if d['state']=='pending' else (50,210,70))
            b = list(map(int, d['xyxy']))
            cv2.rectangle(out, (b[0],b[1]), (b[2],b[3]), color, 2)
            labels = [f"#{d['track_id']} {d['class_name']} {d['confidence']:.2f} {d['direction']}",
                      f"{d['state']} h={d['height_ratio']:.2f} entry={d.get('entry_active',False)}"]
            x = max(0, min(b[0], w-210))
            y = max(95, min(b[1]+16, h-30))
            for line, label in enumerate(labels):
                yy = y+line*16
                size = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0]
                cv2.rectangle(out,(x,yy-13),(min(w-1,x+size[0]+4),yy+3),(20,20,20),-1)
                cv2.putText(out,label,(x+2,yy),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
        cv2.rectangle(out,(0,0),(w,min(h,80)),(25,25,25),-1)
        cv2.putText(out,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        status = (('最近提示：'+prompt) if self.font else ('Last alert: '+prompt)) if prompt else ('未偵測到目標（不代表安全）' if self.font else 'No target / NOT a safety clearance')
        if quality:
            label=f"Image:{quality['state']} Motion:{quality['motion']['state']}"
            cv2.putText(out,label,(8,max(95,h-12)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,230,255),1,cv2.LINE_AA)
        cv2.putText(out,'S:summary  M:mode  Q:quit',(8,73),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1,cv2.LINE_AA)
        if age is not None:
            cv2.putText(out, f'HELD result age={age:.3f}s; not inferred this frame', (8,90),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,255),1,cv2.LINE_AA)
        if self.font:
            from PIL import Image, ImageDraw
            import numpy as np
            im = Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(im).text((8,32),status,font=self.font,fill=(255,230,80))
            return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        cv2.putText(out, status if status.isascii() else 'Last alert in event log / install CJK font',
                    (8,48),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1,cv2.LINE_AA)
        return out
