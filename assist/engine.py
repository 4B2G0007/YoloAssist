"""Small deterministic tracker and rules. All times supplied by caller in seconds."""
from dataclasses import dataclass
from .config import CLASSES, NEW_CLASSES

DIRECTIONS = {'left': '左前方', 'center': '前方', 'right': '右前方'}

def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.

def overlap(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    denom = (a[2]-a[0])*(a[3]-a[1])
    return area / denom if denom > 0 else 0.

def polygon_overlap(box, polygon):
    # Clip a convex polygon by an axis-aligned rectangle.
    points=[tuple(p) for p in polygon]
    for axis,bound,sign in [(0,box[0],1),(0,box[2],-1),(1,box[1],1),(1,box[3],-1)]:
        output=[]
        if not points:break
        previous=points[-1];pin=(previous[axis]-bound)*sign>=0
        for current in points:
            cin=(current[axis]-bound)*sign>=0
            if cin!=pin:
                factor=(bound-previous[axis])/(current[axis]-previous[axis])
                output.append(tuple(previous[k]+factor*(current[k]-previous[k]) for k in [0,1]))
            if cin:output.append(current)
            previous,pin=current,cin
        points=output
    area=abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1])))/2 if points else 0
    denom=(box[2]-box[0])*(box[3]-box[1])
    return area/denom if denom>0 else 0.

@dataclass
class Track:
    id: int
    cls: int
    box: list
    seen: float
    hits: int = 0
    last_qualifying: float = -1e9
    state: str = 'detected'
    outside_since: float | None = None
    outside_hits: int = 0
    origin_side: str | None = None
    inside_since: float | None = None
    inside_hits: int = 0
    entry_active: bool = False

class Engine:
    def __init__(self, config):
        self.c = config
        self.tracks = {}
        self.next_id = 1
        self.next_event = 1
        self.cooldowns = {}
        self.last_global = {'normal': -1e9, 'urgent': -1e9}
        self.mode=config['alert_mode']
        self.scene=[]
        self.next_slot=1
        self.last_summary=-1e9

    def expire(self, now):
        transitions = []
        for tid, t in list(self.tracks.items()):
            if now - t.seen > self.c['track_ttl']:
                transitions.append({'track_id': tid, 'from': t.state, 'to': 'lost', 'reason': 'track_timeout'})
                del self.tracks[tid]
        return transitions

    def set_mode(self, mode):
        if mode not in ['walk','classic']:raise ValueError('Invalid alert mode')
        self.mode=mode
        self.scene.clear()

    def auxiliary(self, now, event_type, text, clip=None, level='normal', priority=0, **extra):
        event=dict(event_id=self.next_event,time_s=now,event_type=event_type,text=text,
                   level=level,priority=priority,decision='selected',reason='auxiliary_event',
                   playback='not_scheduled',track_id=None,direction=None,class_name=None,**extra)
        if clip:event['clip']=clip
        self.next_event+=1
        return event

    def _entry(self, track, row, now, motion_ok):
        if not motion_ok or track.cls in NEW_CLASSES:
            track.outside_since=None;track.origin_side=None;track.inside_since=None
            track.outside_hits=track.inside_hits=0;track.entry_active=False
            return False, 'motion_untrusted' if not motion_ok else 'not_entry_class'
        ov=row['roi_overlap']
        if ov<=self.c['entry_outside_overlap']:
            if track.outside_since is None:track.outside_since=now
            track.outside_hits+=1
            track.inside_since=None;track.inside_hits=0;track.entry_active=False
            if row['direction'] in ['left','right'] and track.outside_hits>=self.c['entry_min_hits'] and now-track.outside_since>=self.c['entry_outside_s']:
                track.origin_side=row['direction']
        elif ov>=self.c['entry_inside_overlap']:
            if track.origin_side:
                if track.inside_since is None:track.inside_since=now
                track.inside_hits+=1
                if (track.inside_hits>=self.c['entry_min_hits'] and now-track.inside_since>=self.c['entry_inside_s']
                    and row['height_ratio']>=self.c['entry_min_height']):track.entry_active=True
        else:
            track.inside_since=None;track.inside_hits=0
        return track.entry_active, ('confirmed_entry' if track.entry_active else 'observing_entry')

    def _slot(self, proposal, box, now):
        self.scene=[slot for slot in self.scene if now-slot['seen']<=self.c['scene_release_s']]
        for slot in self.scene:
            if (slot['class_name']==proposal['class_name'] and slot['direction']==proposal['direction']
                and slot['event_type']==proposal['event_type']
                and (slot['track_id']==proposal['track_id'] or iou(box,slot['box'])>=self.c['track_iou'])):
                slot.update(box=box,seen=now,track_id=proposal['track_id'])
                return slot
        slot=dict(slot_id=self.next_slot,class_name=proposal['class_name'],direction=proposal['direction'],
                  event_type=proposal['event_type'],box=box,track_id=proposal['track_id'],seen=now,announced=None)
        self.next_slot+=1;self.scene.append(slot)
        return slot

    def update(self, candidates, width, height, now, motion_ok=True, perception_ok=True):
        transitions = self.expire(now)
        rows, valid = [], []
        for d in candidates:
            cls, conf = int(d['class_id']), float(d['confidence'])
            raw = list(map(float, d['xyxy']))
            b = [max(0., min(1., raw[0]/width)), max(0., min(1., raw[1]/height)),
                 max(0., min(1., raw[2]/width)), max(0., min(1., raw[3]/height))]
            row = {'class_id': cls, 'class_name': CLASSES.get(cls, (str(cls), '其他'))[0],
                   'confidence': conf, 'xyxy': raw, 'xyxyn': b, 'track_id': None,
                   'eligible': False, 'reason': '', 'selected_event_id': None}
            rows.append(row)
            if cls not in CLASSES:
                row['reason'] = 'non_target_class'
            elif conf < self.c['alert_conf']:
                row['reason'] = 'below_alert_confidence'
            elif b[2] <= b[0] or b[3] <= b[1]:
                row['reason'] = 'invalid_box'
            else:
                row['eligible'] = True
                valid.append(row)
        # Global greedy, one-to-one association. A track cannot match two boxes.
        pairs = []
        for index, r in enumerate(valid):
            for tid, t in self.tracks.items():
                score = iou(r['xyxyn'], t.box)
                if r['class_id'] == t.cls and score >= self.c['track_iou']:
                    pairs.append((score, index, tid))
        assigned, used = {}, set()
        for score, index, tid in sorted(pairs, reverse=True):
            if index not in assigned and tid not in used:
                assigned[index] = tid
                used.add(tid)
        proposals = []
        for index, row in enumerate(valid):
            b = row['xyxyn']
            tid = assigned.get(index)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, row['class_id'], b, now)
            t = self.tracks[tid]
            before = t.state
            if now - t.seen > self.c['max_confirm_gap']:
                t.hits = 0
                t.state = 'detected'
                t.origin_side=None;t.outside_since=None;t.inside_since=None
                t.inside_hits=t.outside_hits=0;t.entry_active=False
            t.box, t.seen = b, now
            center = (b[0]+b[2])/2
            direction = 'left' if center < 1/3 else ('right' if center >= 2/3 else 'center')
            h, ov = b[3]-b[1], overlap(b, self.c['roi'])
            thresholds = self.c['thresholds'][row['class_name']]
            poly_ov=polygon_overlap(b,self.c['near_polygon'])
            in_roi = (poly_ov>=self.c['near_polygon_overlap']) if t.cls in NEW_CLASSES else ov >= self.c['min_overlap']
            qualifies = in_roi and h >= thresholds['near']
            if qualifies:
                t.hits = t.hits + 1 if now - t.last_qualifying <= self.c['max_confirm_gap'] else 1
                t.last_qualifying = now
            else:
                t.hits = 0
            immediate = in_roi and h >= thresholds['immediate']
            if immediate or (qualifies and t.hits >= self.c['confirm_hits']):
                t.state = 'near'
                reason = 'immediate_threshold' if immediate else 'confirmed_near'
            elif before == 'near' and in_roi and h >= thresholds['release'] and now-t.last_qualifying <= self.c['max_confirm_gap']:
                t.state, reason = 'near', 'hysteresis_hold'
            elif qualifies:
                t.state, reason = 'pending', 'waiting_confirmation'
            else:
                t.state, reason = 'detected', 'outside_roi' if not in_roi else 'below_near_threshold'
            row.update(track_id=tid, direction=direction, height_ratio=h, roi_overlap=ov,
                       thresholds=thresholds.copy(), in_roi=in_roi, qualifies=qualifies,
                       consecutive_hits=t.hits, state_before=before, state=t.state, reason=reason,
                       near_polygon_overlap=poly_ov,motion_ok=motion_ok,perception_ok=perception_ok)
            entered, entry_reason=self._entry(t,row,now,motion_ok and perception_ok)
            row.update(entry_active=entered,entry_reason=entry_reason,entry_origin=t.origin_side,
                       entry_outside_hits=t.outside_hits,entry_inside_hits=t.inside_hits)
            if before != t.state:
                transitions.append({'track_id': tid, 'from': before, 'to': t.state, 'reason': reason})
            level = 'urgent' if t.state == 'near' else 'normal'
            spoken_direction='center' if level=='urgent' and in_roi else direction
            name=CLASSES[t.cls][1]
            slug=row['class_name'].replace(' ','_')
            event_type='near' if level=='urgent' else ('entry' if entered else 'detection')
            priority=3 if level=='urgent' else (2 if entered else 1)
            if event_type=='entry':
                side=t.origin_side
                text=f"注意，{'左側' if side=='left' else '右側'}{name}進入前方。"
                clip=f'entry_{side}_{slug}.wav'
                spoken_direction='center'
            else:
                text=(f'注意，{DIRECTIONS[spoken_direction]}有{name}。' if level=='urgent' and t.cls in NEW_CLASSES else
                      (f'注意，{DIRECTIONS[spoken_direction]}{name}靠近。' if level=='urgent' else f'{DIRECTIONS[spoken_direction]}有{name}。'))
                clip=f'{level}_{spoken_direction}_{slug}.wav'
            proposals.append({'track_id':tid,'class_name':row['class_name'],'direction':spoken_direction,
                              'level':level,'text':text,'clip':clip,'event_type':event_type,'priority':priority,
                              'entry_origin':t.origin_side,'also_entry':entered and level=='urgent',
                              'height_ratio':h,'in_roi':in_roi,'rule_score':h+(1 if in_roi else 0),'row':row})
        # Missing detections break consecutive confirmation and immediately suppress a held near state.
        observed = {r['track_id'] for r in valid}
        for tid, t in self.tracks.items():
            if tid not in observed:
                t.hits = 0
                t.origin_side=None;t.outside_since=None;t.inside_since=None
                t.outside_hits=t.inside_hits=0;t.entry_active=False
                if t.state != 'detected':
                    transitions.append({'track_id': tid, 'from': t.state, 'to': 'detected', 'reason': 'not_observed'})
                t.state = 'detected'
        selected, events = None, []
        proposals.sort(key=lambda x: (x['priority'], x['rule_score'], -x['track_id']), reverse=True)
        for p in proposals:
            row = p.pop('row')
            level = p['level']
            key = (p['class_name'], p['direction'], level)
            cd = self.c[level+'_cooldown']
            remaining = max(0., cd - (now - self.cooldowns.get(key, -1e9)))
            gap = self.c[level+'_global_gap']
            slot=self._slot(p,row['xyxyn'],now) if self.mode=='walk' and p['event_type']!='detection' else None
            if not perception_ok:
                decision,why='suppressed','image_quality_degraded'
            elif self.mode=='walk' and p['event_type']=='detection':
                decision,why='suppressed','walking_background'
            elif slot is not None and slot['announced'] is not None and (p['level']!='urgent' or now-slot['announced']<self.c['walk_urgent_repeat_s']):
                decision,why='suppressed','scene_already_announced'
            elif selected:
                decision, why = 'suppressed', 'lower_priority'
            elif remaining > 0:
                decision, why = 'suppressed', 'cooldown'
            elif now-self.last_global[level] < gap:
                decision, why = 'suppressed', 'global_rate_limit'
            else:
                decision, why = 'selected', 'highest_available_priority'
            eid = self.next_event
            self.next_event += 1
            event = dict(p, event_id=eid, time_s=now, decision=decision, reason=why,
                         cooldown_remaining_s=remaining, playback='not_scheduled',
                         alert_mode=self.mode,scene_slot_id=slot['slot_id'] if slot else None)
            events.append(event)
            row['event_id'] = eid
            row['prompt_decision'] = decision
            row['prompt_reason'] = why
            if decision == 'selected':
                selected = event
                row['selected_event_id'] = eid
                if slot is not None:slot['announced']=now
                self.cooldowns[key] = now
                self.last_global[level] = now
        return rows, events, selected, transitions
