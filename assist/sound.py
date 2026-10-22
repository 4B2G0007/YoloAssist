"""Bounded clip playback with priority, explicit missing-voice status, summary cancellation."""
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from .voice_catalog import event_clips

class AudioPlayer:
    def __init__(self, directory, enabled=True, max_age=2.5):
        self.directory=Path(directory);self.enabled=enabled;self.max_age=max_age
        self.pending=None;self.active=None;self.cancel_generation=0
        self.cv=threading.Condition();self.stopping=False;self.events=queue.Queue()
        self.backend=('winsound' if sys.platform=='win32' else
                      next((x for x in ['aplay','paplay','afplay','ffplay'] if shutil.which(x)),None))
        self.thread=None
        if enabled:
            self.thread=threading.Thread(target=self._work,daemon=True);self.thread.start()

    @staticmethod
    def priority(event):
        return event.get('priority',3 if event.get('level')=='urgent' else 1)

    def emit(self,event,status,**extra):
        self.events.put(dict(event_id=event['event_id'],time_s=event['time_s'],
                             frame_index=event.get('frame_index'),kind='playback',playback=status,
                             event_type=event.get('event_type'),monotonic_s=time.monotonic(),**extra))

    def submit(self,event):
        if not self.enabled:
            self.emit(event,'not_played',reason='analysis_or_no_audio');return
        # Preflight all clips so a partial description is never reported as complete.
        names=event_clips(event)
        missing=[n for n in names if not (self.directory/n).is_file()]
        if missing:
            self.emit(event,'failed',reason='missing_voice',clips=missing)
            print(f'[Audio unavailable] {", ".join(missing)}',file=sys.stderr);return
        if not names:
            self.emit(event,'not_played',reason='no_spoken_content');return
        with self.cv:
            p=self.priority(event)
            if event.get('event_type')=='summary' and self.active and self.priority(self.active)>p:
                self.emit(event,'dropped',reason='higher_priority_active');return
            if self.pending:
                if self.priority(self.pending[0])>p:
                    self.emit(event,'dropped',reason='higher_priority_pending');return
                self.emit(self.pending[0],'dropped',reason='replaced_by_newer_event')
            if self.active and self.active.get('event_type')=='summary' and p>self.priority(self.active):
                self.cancel_generation+=1
            self.pending=(dict(event),time.monotonic())
            self.emit(event,'queued');self.cv.notify()

    def cancel_pending(self,reason='perception_unavailable'):
        with self.cv:
            self.cancel_generation+=1
            if self.pending:
                self.emit(self.pending[0],'dropped',reason=reason);self.pending=None

    def _play(self,path):
        if self.backend=='winsound':
            import winsound
            winsound.PlaySound(str(path),winsound.SND_FILENAME|winsound.SND_NODEFAULT)
        elif self.backend:
            args={'aplay':['aplay','-q'],'paplay':['paplay'],'afplay':['afplay'],
                  'ffplay':['ffplay','-nodisp','-autoexit','-loglevel','error']}[self.backend]
            result=subprocess.run(args+[str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=15)
            if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-400:])
        else:raise RuntimeError('No audio player available; Linux: install/configure alsa-utils or ffmpeg.')

    def _work(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.pending is not None or self.stopping)
                if self.stopping:return
                event,queued=self.pending;self.pending=None;self.active=event
                generation=self.cancel_generation
            try:
                if time.monotonic()-queued>self.max_age:
                    self.emit(event,'dropped',reason='expired');continue
                names=event_clips(event)
                self.emit(event,'started',clips=names,queue_delay_s=time.monotonic()-queued)
                complete=True
                for index,name in enumerate(names):
                    with self.cv:
                        cancel=(self.stopping or generation!=self.cancel_generation)
                    if cancel:
                        self.emit(event,'dropped',reason='cancelled_between_clips');complete=False;break
                    self.emit(event,'clip_started',clip=name,clip_index=index)
                    self._play(self.directory/name)
                    self.emit(event,'clip_finished',clip=name,clip_index=index)
                if complete:self.emit(event,'finished')
            except Exception as e:
                self.emit(event,'failed',reason=str(e));print(f'[Audio warning] {e}',file=sys.stderr)
            finally:
                with self.cv:self.active=None

    def drain(self):
        result=[]
        while True:
            try:result.append(self.events.get_nowait())
            except queue.Empty:return result

    def close(self):
        with self.cv:
            self.stopping=True;self.cancel_generation+=1
            if self.pending:self.emit(self.pending[0],'dropped',reason='run_finished');self.pending=None
            self.cv.notify()
        if self.thread:self.thread.join(timeout=16)
