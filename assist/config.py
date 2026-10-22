"""Validated, snapshot-friendly configuration. Ratios are heuristics, not metres."""
import copy
import json
import math
from pathlib import Path

CLASSES = {0: ('person', '行人'), 1: ('bicycle', '自行車'), 2: ('car', '汽車'),
           3: ('motorcycle', '機車'), 5: ('bus', '公車'), 7: ('truck', '卡車'),
           13: ('bench', '長椅'), 16: ('dog', '狗'), 28: ('suitcase', '行李箱'),
           56: ('chair', '椅子'), 58: ('potted plant', '盆栽')}
NEW_CLASSES = {13, 16, 28, 56, 58}
DEFAULT = {
    'model': 'yolo11n.pt', 'device': 'auto', 'imgsz': 640,
    'model_conf': 0.15, 'alert_conf': 0.45, 'nms_iou': 0.45,
    'roi': [0.3, 0.0, 0.7, 1.0], 'min_overlap': 0.25,
    'confirm_hits': 2, 'max_confirm_gap': 1.5, 'track_ttl': 1.5, 'track_iou': 0.2,
    'normal_cooldown': 3.0, 'urgent_cooldown': 2.0,
    'normal_global_gap': 1.0, 'urgent_global_gap': 0.7,
    'audio_max_age': 2.5, 'stale_seconds': 2.0,
    'stride': 1, 'camera_size': [640, 480], 'camera_timeout': 5.0,
    'camera_index': 0, 'camera_format': 'RGB888', 'camera_fps': 30.0,
    'camera_autofocus': 'continuous', 'camera_warmup_s': 1.0,
    'font': '', 'font_size': 20,
    'alert_mode': 'walk',
    'near_polygon': [[0.40,0.45],[0.60,0.45],[0.85,1.0],[0.15,1.0]],
    'near_polygon_overlap': 0.18,
    'entry_outside_overlap': 0.08, 'entry_inside_overlap': 0.30,
    'entry_outside_s': 0.20, 'entry_inside_s': 0.20, 'entry_min_height': 0.18,
    'entry_min_hits': 2, 'scene_release_s': 2.0, 'walk_urgent_repeat_s': 12.0,
    'summary_max_items': 3, 'summary_cooldown_s': 3.0,
    'quality_hold_s': 1.0, 'quality_recover_s': 0.7, 'quality_repeat_s': 20.0,
    'quality_dark_mean': 25.0, 'quality_dark_pixel': 35, 'quality_dark_ratio': 0.85,
    'quality_detail_std': 8.0, 'quality_detail_lap': 8.0,
    'quality_blur_reference': 80.0, 'quality_blur_lap': 40.0, 'quality_blur_ratio': 0.25,
    'motion_max_gap': 1.5, 'motion_min_features': 12, 'motion_min_inlier_ratio': 0.65,
    'motion_speed': 0.22, 'motion_rotation': 15.0, 'motion_scale': 0.20,
    'thresholds': {
        'person': {'near': 0.45, 'immediate': 0.75, 'release': 0.38},
        'bicycle': {'near': 0.35, 'immediate': 0.65, 'release': 0.28},
        'car': {'near': 0.35, 'immediate': 0.65, 'release': 0.28},
        'motorcycle': {'near': 0.40, 'immediate': 0.70, 'release': 0.32},
        'bus': {'near': 0.50, 'immediate': 0.80, 'release': 0.42},
        'truck': {'near': 0.50, 'immediate': 0.80, 'release': 0.42},
        'bench': {'near':0.30,'immediate':0.65,'release':0.24},
        'dog': {'near':0.30,'immediate':0.65,'release':0.24},
        'suitcase': {'near':0.30,'immediate':0.65,'release':0.24},
        'chair': {'near':0.35,'immediate':0.65,'release':0.28},
        'potted plant': {'near':0.35,'immediate':0.65,'release':0.28}}}

def validate(c):
    def number(k, low, high=None, integer=False):
        x = c[k]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f'{k}: must be a finite number')
        if x < low or (high is not None and x > high) or (integer and type(x) is not int):
            raise ValueError(f'{k}: invalid value {x}')
    for k in ['model_conf', 'alert_conf', 'nms_iou', 'min_overlap', 'track_iou',
              'near_polygon_overlap','entry_outside_overlap','entry_inside_overlap','entry_min_height',
              'quality_dark_ratio','quality_blur_ratio','motion_min_inlier_ratio']:
        number(k, 0, 1)
    if c['model_conf'] > c['alert_conf']:
        raise ValueError('model_conf must not exceed alert_conf')
    for k in ['max_confirm_gap', 'track_ttl', 'normal_cooldown', 'urgent_cooldown',
              'normal_global_gap', 'urgent_global_gap', 'audio_max_age', 'stale_seconds', 'camera_timeout',
              'entry_outside_s','entry_inside_s','scene_release_s','walk_urgent_repeat_s','summary_cooldown_s',
              'quality_hold_s','quality_recover_s','quality_repeat_s','motion_max_gap','motion_speed',
              'motion_rotation','motion_scale','quality_dark_mean','quality_detail_std','quality_detail_lap',
              'quality_blur_reference','quality_blur_lap']:
        number(k, 0.01)
    for k in ['stride', 'confirm_hits', 'imgsz', 'font_size','entry_min_hits','summary_max_items','motion_min_features','quality_dark_pixel']:
        number(k, 1, integer=True)
    number('camera_index', 0, integer=True)
    number('camera_fps', 1, 120)
    number('camera_warmup_s', 0, 10)
    if c['camera_format'] not in ['XBGR8888','XRGB8888','RGB888','BGR888']:
        raise ValueError('Unsupported Picamera2 camera_format')
    if c['camera_autofocus'] not in ['continuous','default']:
        raise ValueError('camera_autofocus must be continuous or default')
    if c['alert_mode'] not in ['walk','classic']:
        raise ValueError('alert_mode must be walk or classic')
    if c['summary_max_items']>3 or c['quality_dark_pixel']>255:
        raise ValueError('summary_max_items <= 3 and quality_dark_pixel <= 255 required')
    if c['entry_outside_overlap']>=c['entry_inside_overlap']:
        raise ValueError('entry outside overlap must be less than inside overlap')
    poly=c['near_polygon']
    if len(poly)!=4 or any(len(p)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p) for p in poly):
        raise ValueError('near_polygon must have four normalized points')
    # Convex, ordered quadrilateral required by clipping implementation.
    turns=[]
    for i in range(4):
        a,b,d=poly[i],poly[(i+1)%4],poly[(i+2)%4]
        turns.append((b[0]-a[0])*(d[1]-b[1])-(b[1]-a[1])*(d[0]-b[0]))
    if not (all(t>0 for t in turns) or all(t<0 for t in turns)):
        raise ValueError('near_polygon must be a strictly convex ordered quadrilateral')
    r = c['roi']
    if len(r) != 4 or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1):
        raise ValueError('roi must be normalized [left, top, right, bottom]')
    if len(c['camera_size']) != 2 or any(type(x) is not int or x < 1 for x in c['camera_size']):
        raise ValueError('camera_size must contain two positive integers')
    for name, _ in CLASSES.values():
        t = c['thresholds'][name]
        if not 0 < t['release'] < t['near'] < t['immediate'] <= 1:
            raise ValueError(f'{name}: require 0 < release < near < immediate <= 1')
    return c

def load(path=None):
    c = copy.deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        unknown = set(supplied) - set(c)
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        for k, v in supplied.items():
            if k == 'thresholds':
                if set(v) - set(c[k]):
                    raise ValueError('Unknown threshold class')
                for name, t in v.items():
                    if set(t) - set(c[k][name]):
                        raise ValueError(f'Unknown threshold key: {name}')
                    c[k][name].update(t)
            else:
                c[k] = v
    return validate(c)
