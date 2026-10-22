import math
import time
from collections import Counter
from pathlib import Path

from .engine import Engine
from .recording import Recorder, sha256
from .render import Overlay
from .sound import AudioPlayer
from .sources import LatestCamera
from .quality import QualityMonitor
from .interaction import make_summary
from .voice_catalog import additions, event_clips


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, math.ceil(len(ordered)*p)-1)]


def run(source, config, detector, output='results', record=True, display=False,
        realtime=False, audio=False, audio_dir='audio', max_frames=None, summary_at=None, key_provider=None):
    import cv2
    is_camera = str(source).isdigit() or str(source)=='picamera2'
    stem = ('picamera2' if str(source)=='picamera2' else 'camera'+str(source)) if is_camera else Path(source).stem
    # A per-run engine/player prevents state leaking across files.
    rec = Recorder(output, stem, record, config, {
        'source': str(source), 'source_sha256': None if is_camera else sha256(source),
        'model_sha256': sha256(config['model']), 'resolved_device': getattr(detector,'device','unknown'),
        'mode': 'camera' if is_camera else ('video_preview' if realtime else 'video_analysis'),
        'audio_requested': audio, 'display_requested': display, 'summary_at':summary_at or [],
        'missing_new_voices':[name for name in additions() if not (Path(audio_dir)/name).is_file()],
        'time_basis': 'monotonic_since_start' if is_camera else 'source_frame_index/fps',
        'video_audio': 'silent_output_no_original_audio',
        'candidate_scope': 'post_model_confidence_and_NMS_all_classes'})
    missing_voices=[name for name in additions() if not (Path(audio_dir)/name).is_file()]
    print(f'[{stem}] '+(f'Output: {rec.directory}' if record else 'Recording OFF'))
    engine, overlay = Engine(config), Overlay(config)
    monitor=QualityMonitor(config)
    scheduled=sorted(summary_at or [])
    quality={'state':'initializing','usable':False,'motion_ok':False,'motion':{'state':'unknown'}}
    quality_counts,motion_counts,event_counts=Counter(),Counter(),Counter()
    player = AudioPlayer(audio_dir, enabled=audio, max_age=config['audio_max_age'])
    cap = cam = writer = None
    started = time.monotonic()
    last_progress = started
    status, error = 'failed', None
    read_count = inferred = skipped = dropped = 0
    frame_index = -1
    fps, total, duration = None, None, None
    latencies, infer_ms = [], []
    observations, prompts, audio_states = Counter(), Counter(), Counter()
    rows, prompt, last_inferred_time, last_inferred_index = [], '', None, None
    raw_time = 0.
    last_result_usable=False
    video_path = rec.directory/f'{stem}_annotated.mp4'
    video_written = 0
    warnings = []
    prior_serial = 0
    cancelled = False

    def audio_log():
        ev = player.drain()
        for e in ev:
            audio_states[e['playback']] += 1
        rec.events(ev)

    def deliver(event):
        nonlocal prompt
        prompt=event['text']
        if realtime and not is_camera and time.monotonic()-started-raw_time>config['audio_max_age']:
            rec.events([{'kind':'playback','event_id':event['event_id'],'frame_index':frame_index,
                         'time_s':raw_time,'playback':'dropped','reason':'preview_lag'}])
            audio_states['dropped_preview_lag']+=1
        else:player.submit(event)

    def log_aux(event):
        event['voice_files']=event_clips(event)
        event['voice_available']=all((Path(audio_dir)/name).is_file() for name in event['voice_files'])
        event.update(kind='decision',frame_index=frame_index,monotonic_s=time.monotonic(),
                     quality_state=quality['state'],motion_state=quality['motion']['state'],alert_mode=engine.mode)
        rec.events([event])
        event_counts[event['event_type']+'_'+event['decision']]+=1
        if event['decision']=='selected':deliver(event)

    def request_summary():
        valid=(last_result_usable and not stale and last_inferred_time is not None and raw_time-last_inferred_time<=config['stale_seconds'])
        event=make_summary(engine,rows,raw_time,valid,quality_ok=quality['usable'])
        print('[Summary] '+(event['text'] or event['reason']),flush=True)
        log_aux(event)

    try:
        if is_camera:
            if str(source)=='picamera2':
                cam = LatestCamera(source, config['camera_size'], options=config)
            else:
                cam = LatestCamera(source, config['camera_size'])
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video: {source}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise RuntimeError('Video FPS is unavailable; convert to a constant-frame-rate MP4 first.')
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total = int(n) if math.isfinite(n) and n > 0 else None
            duration = total/fps if total else None
        while True:
            if max_frames and read_count >= max_frames:
                status, cancelled = 'limited', True
                break
            if is_camera:
                serial, frame, acquired = cam.read(prior_serial, config['camera_timeout'])
                dropped += max(0, serial-prior_serial-1)
                prior_serial = serial
                frame_index = serial-1
                raw_time = acquired-started
            else:
                ok, frame = cap.read()
                if not ok:
                    if total is not None and read_count < total:
                        raise RuntimeError(f'Video read stopped early: {read_count}/{total} frames')
                    status = 'completed' if total else 'ended_unverified'
                    if total is None:
                        warnings.append('No frame count metadata: EOF vs decode failure cannot be fully distinguished.')
                    break
                frame_index += 1
                raw_time = frame_index/fps
                acquired = time.monotonic()
            read_count += 1
            h,w = frame.shape[:2]
            # Camera recording is diagnostic: nominal FPS follows source timestamps only approximately.
            if record and writer is None:
                out_fps = fps if fps else 10.
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w,h))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create annotated video: {video_path}')
                if is_camera:
                    warnings.append('Camera MP4 is nominal 10 FPS, not wall-time-faithful; use per-frame JSON timestamps.')
            do_infer = (read_count-1) % config['stride'] == 0
            t0 = time.monotonic()
            stale = False
            decisions, transitions = [], []
            elapsed_infer_ms = None
            if do_infer:
                candidates = detector.predict(frame)
                elapsed_infer_ms = (time.monotonic()-t0)*1000
                infer_ms.append(elapsed_infer_ms)
                inferred += 1
                quality=monitor.check(frame,raw_time,candidates)
                quality_counts[quality['state']]+=1
                motion_counts[quality['motion']['state']]+=1
                decision_time = time.monotonic()
                latency = decision_time-acquired
                latencies.append(latency*1000)
                stale = is_camera and latency > config['stale_seconds']
                last_result_usable=not stale
                if stale:
                    rows, decisions, chosen = [], [], None
                    player.cancel_pending('stale_inference')
                    transitions = engine.expire(raw_time+latency)
                    rec.events([{'kind':'error','frame_index':frame_index,'time_s':raw_time,
                                 'reason':'stale_inference_result_suppressed'}])
                    prompt = '影像處理延遲過大，警示暫停' if overlay.font else 'STALE IMAGE: alerts suppressed'
                else:
                    rows, decisions, chosen, transitions = engine.update(candidates,w,h,raw_time,motion_ok=quality['motion_ok'],perception_ok=quality['usable'])
                    observations.update(r['class_name'] for r in rows if r['eligible'])
                    for e in decisions:
                        e['voice_files']=event_clips(e)
                        e['voice_available']=all((Path(audio_dir)/name).is_file() for name in e['voice_files'])
                        e.update(kind='decision', frame_index=frame_index, playback='not_scheduled',
                                 monotonic_s=decision_time,quality_state=quality['state'],motion_state=quality['motion']['state'])
                    rec.events(decisions)
                    for e in decisions:
                        prompts[(e['decision'],e['level'])] += 1
                        event_counts[e['event_type']+'_'+e['decision']]+=1
                    if not quality['usable']:player.cancel_pending('image_quality_degraded')
                    if chosen:deliver(chosen)
                    elif not any(r['eligible'] for r in rows):prompt=''
                    if quality.get('notice'):
                        notice=quality['notice'];clip=notice+'.wav'
                        event=engine.auxiliary(raw_time,notice,additions()[clip],clip=clip,
                                               level='urgent' if notice!='quality_recovered' else 'normal',
                                               priority=4 if notice!='quality_recovered' else 0)
                        if chosen and notice=='quality_recovered':
                            event.update(decision='suppressed',reason='higher_priority_object_event')
                        log_aux(event)
                last_inferred_time, last_inferred_index = raw_time, frame_index
            else:
                skipped += 1
                candidates = None
                if last_inferred_time is not None and raw_time-last_inferred_time > config['stale_seconds']:
                    rows, prompt = [], ''
                    transitions = engine.expire(raw_time)
            while scheduled and raw_time>=scheduled[0]:
                scheduled.pop(0)
                request_summary()
            audio_log()
            preview_lag = max(0., time.monotonic()-started-raw_time) if realtime and not is_camera else None
            metadata = {'frame_index':frame_index, 'source_time_s':raw_time, 'source_time_ms':raw_time*1000,
                        'width':w,'height':h,'inferred':do_infer,'stale_result':stale,
                        'inference_ms':elapsed_infer_ms,'last_inferred_frame':last_inferred_index,
                        'result_age_s': None if last_inferred_time is None else raw_time-last_inferred_time,
                        'candidates':candidates,'objects':rows if do_infer else [],
                        'transitions':transitions, 'decisions':decisions,
                        'acquired_monotonic_s':acquired, 'processed_monotonic_s':time.monotonic(),
                        'preview_lag_s':preview_lag,'quality':quality,'alert_mode':engine.mode}
            rec.inference(metadata)
            if transitions:
                rec.events([dict(kind='transition',frame_index=frame_index,time_s=raw_time,
                                 track_id=t['track_id'],reason=f"{t['from']}->{t['to']}: {t['reason']}") for t in transitions])
            if writer is not None or display:
                mean_ms = sum(infer_ms)/len(infer_ms) if infer_ms else 0
                title = f't={raw_time:.2f}s #{frame_index} {mean_ms:.0f}ms {engine.mode}'
                drawn = overlay.draw(frame,rows,title,prompt,
                                     age=(raw_time-last_inferred_time) if not do_infer and last_inferred_time is not None else None,quality=quality)
                if writer is not None:
                    writer.write(drawn)
                    video_written += 1
                if display:
                    cv2.imshow('YOLO Assist V1',drawn)
            key=(key_provider(read_count) if key_provider else (cv2.waitKey(1)&0xff if display else -1))
            if key in [ord('q'),27]:
                status,cancelled='interrupted',True
                break
            if key in [ord('s'),ord('S')]:request_summary()
            if key in [ord('m'),ord('M')]:
                engine.set_mode('classic' if engine.mode=='walk' else 'walk')
                rec.events([{'kind':'mode_change','frame_index':frame_index,'time_s':raw_time,'alert_mode':engine.mode}])
                print('[Mode] '+engine.mode,flush=True)
            if realtime and not is_camera:
                delay = started+(frame_index+1)/fps-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if time.monotonic()-last_progress > 5:
                print(f'[{stem}] read={read_count}/{total or "?"}, inferred={inferred}, t={raw_time:.2f}s',flush=True)
                rec.summary({'status':'running','frames_read':read_count,'frames_inferred':inferred,
                             'last_source_time_s':raw_time,'expected_frames':total})
                last_progress = time.monotonic()
    except KeyboardInterrupt:
        status, error, cancelled = 'interrupted', 'KeyboardInterrupt', True
    except Exception as e:
        error = f'{type(e).__name__}: {e}'
        status = 'failed'
        print(f'[{stem}] ERROR: {error}')
        rec.events([{'kind':'error','frame_index':frame_index,'time_s':raw_time,'reason':error}])
        if is_camera:
            event = {'event_id':-1,'time_s':raw_time,'frame_index':frame_index,'kind':'camera_error',
                     'level':'urgent','text':'影像中斷，請檢查攝影機。'}
            player.submit(event)
            # Give the worker a chance to dequeue the fault announcement before shutdown.
            time.sleep(0.05)
    finally:
        if cap is not None:
            cap.release()
        if cam is not None:
            cam.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        player.close()
        audio_log()
        wall = time.monotonic()-started
        verified_video_frames = 0
        if writer is not None and video_path.is_file():
            check = cv2.VideoCapture(str(video_path))
            while True:
                ok,_ = check.read()
                if not ok:
                    break
                verified_video_frames += 1
            check.release()
            if verified_video_frames != video_written:
                warnings.append('Annotated video verification frame count mismatch')
                status = 'failed'
                error = error or 'Annotated video is incomplete'
        result = {'status':status,'error':error,'warnings':warnings,'source':str(source),
                  'source_fps':fps,'source_duration_s':duration,'expected_frames':total,
                  'frames_read':read_count,'frames_inferred':inferred,'frames_skipped':skipped,
                  'camera_frames_dropped_before_processing':dropped,'last_frame_index':frame_index,
                  'camera_configuration':getattr(cam,'info',None),
                  'last_source_time_s':raw_time,'wall_seconds':wall,
                  'offline_inferences_per_wall_second': inferred/wall if wall else None,
                  'average_inference_ms':sum(infer_ms)/len(infer_ms) if infer_ms else None,
                  'p95_inference_ms':percentile(infer_ms,.95),
                  'mean_acquisition_to_decision_ms':sum(latencies)/len(latencies) if latencies else None,
                  'p95_acquisition_to_decision_ms':percentile(latencies,.95),
                  'detection_observations':dict(observations),
                  'prompt_counts':{f'{a}_{b}':v for (a,b),v in prompts.items()},
                  'audio_status_counts':dict(audio_states),'event_counts':dict(event_counts),
                  'quality_observations':dict(quality_counts),'motion_observations':dict(motion_counts),
                  'final_alert_mode':engine.mode,'missing_new_voices':missing_voices,
                  'video':{'path':str(video_path) if record else None,'written_frames':video_written,
                           'verified_frames':verified_video_frames,'audio_track':False},
                  'cancel_batch':cancelled, 'output_directory':str(rec.directory) if record else None}
        rec.summary(result)
        rec.close()
    print(f'[{stem}] {status}: read {read_count}, inferred {inferred}')
    return result
