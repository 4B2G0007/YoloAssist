import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from . import __version__

FIELDS = ['kind', 'event_id', 'frame_index', 'time_s', 'track_id', 'class_name', 'direction',
          'level', 'text', 'decision', 'reason', 'cooldown_remaining_s', 'playback', 'monotonic_s',
          'queue_delay_s', 'clip', 'rule_score', 'height_ratio', 'in_roi',
          'event_type','priority','clips','entry_origin','also_entry','alert_mode','scene_slot_id',
          'quality_state','motion_state','summary_items','clip_index','voice_files','voice_available']

def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def sha256(path):
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def versions():
    packages = {}
    for name in ['ultralytics', 'torch', 'opencv-python', 'opencv-python-headless', 'numpy', 'Pillow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': sys.version, 'os': platform.platform(), 'packages': packages}

class Recorder:
    def __init__(self, root, stem, enabled, config, metadata):
        self.enabled = enabled
        self.run_id = uuid.uuid4().hex[:10]
        self.directory = Path(root) / (stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+self.run_id)
        self.stem = stem
        self.files = {}
        self.jf = self.ef = None
        self.flush_time = time.monotonic()
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=False)
            dump(self.directory/'run_config.json', {
                'schema_version': '1.1', 'program_version': __version__, 'run_id': self.run_id,
                'started_at': datetime.now().astimezone().isoformat(),
                'configuration': config, 'environment': versions(), **metadata})
            self.jf = (self.directory/f'{stem}_detections.jsonl').open('w', encoding='utf-8')
            self.ef = (self.directory/f'{stem}_events.csv').open('w', encoding='utf-8-sig', newline='')
            self.csv = csv.DictWriter(self.ef, fieldnames=FIELDS, extrasaction='ignore')
            self.csv.writeheader()
            with (self.directory/'review_notes.csv').open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(['start_time', 'end_time', 'problem', 'expected_behavior'])
            self.summary({'status': 'running'})

    def inference(self, row):
        if self.enabled:
            self.jf.write(json.dumps(dict(run_id=self.run_id, **row), ensure_ascii=False, allow_nan=False)+'\n')
            self.flush()

    def events(self, events):
        if self.enabled:
            for e in events:
                self.csv.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in e.items()})
            self.flush()

    def flush(self, force=False):
        if self.enabled and (force or time.monotonic()-self.flush_time >= 1):
            self.jf.flush()
            self.ef.flush()
            self.flush_time = time.monotonic()

    def summary(self, summary):
        if self.enabled:
            dump(self.directory/f'{self.stem}_summary.json', dict(run_id=self.run_id, **summary))

    def close(self):
        if self.enabled:
            self.flush(True)
            self.jf.close()
            self.ef.close()
