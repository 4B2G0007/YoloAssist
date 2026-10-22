"""Windows / Raspberry Pi entry point. Run from the extracted project directory."""
import argparse
import json
import re
import sys
from pathlib import Path
from assist.config import load, validate
from assist.recording import dump

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.mp4','.avi','.mov','.mkv','.m4v','.webm'}

def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',path.name)]

def is_raspberry_pi():
    """Do not mistake every Linux/ARM computer for a Raspberry Pi."""
    if not sys.platform.startswith('linux'):
        return False
    for path in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        try:
            if 'raspberry pi' in Path(path).read_text(errors='ignore').lower():
                return True
        except OSError:
            pass
    return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='YOLO11 Nano offline street-object warning prototype')
    p.add_argument('--source',help='video / folder / USB camera index / picamera2; on Raspberry Pi defaults to picamera2')
    p.add_argument('--config',help='default: config.pi.json on Raspberry Pi or for picamera2; otherwise config.v1.1.json')
    p.add_argument('--mode',choices=['walk','classic'],help='walk is quiet; classic retains V1 normal reminders')
    p.add_argument('--summary-at',type=float,action='append',default=[],help='Request description at source seconds; repeatable')
    p.add_argument('--model',help='Path to your existing yolo11n.pt')
    p.add_argument('--device',help='auto, cpu or CUDA index such as 0')
    p.add_argument('--camera-index',type=int,help='Picamera2 camera number; default 0')
    p.add_argument('--camera-format',choices=['XBGR8888','XRGB8888','RGB888','BGR888'],help='Picamera2 pixel format')
    p.add_argument('--camera-fps',type=float,help='requested camera capture FPS, NOT YOLO inference FPS')
    p.add_argument('--autofocus',choices=['continuous','default'],help='continuous for Camera Module 3; default leaves camera defaults')
    p.add_argument('--output',default=str(ROOT/'results'))
    p.add_argument('--stride',type=int,help='Infer every N frames; default 1')
    p.add_argument('--show',action='store_true',help='Open preview; press Q or Esc to stop')
    p.add_argument('--realtime',action='store_true',help='Pace a video to original FPS (cannot speed up a slow device)')
    p.add_argument('--audio',action='store_true',help='Play WAV alerts; for a file requires --realtime')
    p.add_argument('--audio-dir',default=str(ROOT/'audio'))
    p.add_argument('--no-record',action='store_true',help='Do not save per-source logs/video')
    p.add_argument('--max-frames',type=int,help='Stop early for a short smoke test; marked limited')
    return p.parse_args(argv)


def configure(args, raspberry_pi=None):
    """Resolve platform defaults without importing torch or opening a camera."""
    raspberry_pi = is_raspberry_pi() if raspberry_pi is None else raspberry_pi
    source = args.source or ('picamera2' if raspberry_pi else None)
    if not source:
        raise ValueError('Please specify --source street.mp4 (desktop), or --source picamera2 (Raspberry Pi camera).')
    config_path = args.config or str(ROOT/('config.pi.json' if raspberry_pi or source=='picamera2' else 'config.v1.1.json'))
    c = load(config_path)
    if args.model:
        c['model'] = str(Path(args.model).resolve())
    elif not Path(c['model']).is_absolute():
        c['model'] = str(ROOT/c['model'])
    if raspberry_pi and args.device not in (None,'auto','cpu'):
        raise ValueError('Raspberry Pi uses CPU, not NVIDIA CUDA. Use --device cpu.')
    if args.device:
        c['device'] = args.device
    elif raspberry_pi:
        c['device'] = 'cpu'
    for attr, key in [('stride','stride'), ('mode','alert_mode'),
                      ('camera_index','camera_index'), ('camera_format','camera_format'),
                      ('camera_fps','camera_fps'), ('autofocus','camera_autofocus')]:
        value = getattr(args, attr)
        if value is not None:
            c[key] = value
    if any(not __import__('math').isfinite(t) or t<0 for t in args.summary_at):
        raise ValueError('summary-at must be finite and nonnegative')
    validate(c)
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('--max-frames must be positive')
    if args.audio and not (source.isdigit() or source=='picamera2') and not args.realtime:
        raise ValueError('For video audio use --realtime --audio; analysis mode logs decisions without playback.')
    return c, source, config_path

def main():
    args = parse_args()
    try:
        c, source, config_path = configure(args)
        print(f'[Startup] source={source}; device={c["device"]}; config={config_path}',flush=True)
        camera = source.isdigit() or source=='picamera2'
        if source=='picamera2':
            print(f'[Camera] Picamera2 #{c["camera_index"]}; {c["camera_size"]}; '
                  f'{c["camera_format"]} -> BGR; capture FPS={c["camera_fps"]}; '
                  f'autofocus={c["camera_autofocus"]}. Stop stream.py/rpicam before starting.',flush=True)
        if camera:
            sources = [source]
        else:
            src = Path(source).resolve()
            if src.is_dir():
                sources = sorted((p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=natural_key)
                if not sources:
                    raise ValueError('No supported video files found in folder (subfolders are not scanned).')
            elif src.is_file() and src.suffix.lower() in EXTENSIONS:
                sources = [src]
            else:
                raise ValueError(f'Video source missing or unsupported: {src} (MP3 is not video)')
        from assist.detector import Detector
        from assist.runner import run
        detector = Detector(c)
        reports = []
        for source in sources:
            report = run(source,c,detector,output=args.output,record=not args.no_record,
                         display=args.show,realtime=args.realtime,audio=args.audio,
                         audio_dir=args.audio_dir,max_frames=args.max_frames,summary_at=args.summary_at)
            reports.append(report)
            if report['cancel_batch']:
                break
        if len(sources)>1 and not args.no_record:
            import uuid
            from datetime import datetime
            target = Path(args.output)/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
            target.parent.mkdir(parents=True,exist_ok=True)
            dump(target,{'requested_files':len(sources),'processed_files':len(reports),'results':reports})
            print(f'Batch summary: {target}')
        return 1 if any(r['status'] in ['failed','ended_unverified'] for r in reports) else 0
    except KeyboardInterrupt:
        print('Interrupted.',file=sys.stderr)
        return 130
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(main())
