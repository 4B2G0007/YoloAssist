"""Check the selected Python environment; --capture additionally opens the CSI camera."""
import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',action='store_true',help='capture one BGR frame without loading YOLO weights')
    parser.add_argument('--config',default=str(ROOT/'config.pi.json'))
    args = parser.parse_args()
    from assist.config import load
    c = load(args.config)
    report = {'python':sys.executable, 'python_version':sys.version,
              'architecture':platform.machine(), 'modules':{},
              'model_exists':(ROOT/c['model']).is_file(),
              'audio_player':next((x for x in ('aplay','paplay','ffplay') if shutil.which(x)),None),
              'capture':None}
    failed = not report['model_exists']
    for name in ('numpy','cv2','torch','ultralytics','picamera2','libcamera'):
        try:
            module = importlib.import_module(name)
            report['modules'][name] = {'ok':True,'version':str(getattr(module,'__version__','system package'))}
        except Exception as exc:
            failed = True
            report['modules'][name] = {'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    if args.capture:
        if failed:
            report['capture'] = {'ok':False,'error':'Fix missing modules/model first.'}
        else:
            from assist.sources import LatestCamera
            cam = LatestCamera('picamera2',c['camera_size'],options=c)
            try:
                _, frame, _ = cam.read(0,c['camera_timeout'])
                report['capture'] = {'ok':True,'shape':list(frame.shape),
                                     'dtype':str(frame.dtype),'color_order':'BGR','settings':cam.info}
            except Exception as exc:
                failed = True
                report['capture'] = {'ok':False,'error':f'{type(exc).__name__}: {exc}'}
            finally:
                cam.close()
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if failed:
        print('Use the same Raspberry Pi OS Python/venv as the camera test; '
              'Picamera2/libcamera should come from apt and --system-site-packages.',file=sys.stderr)
    if report['audio_player'] is None:
        print('Audio warning: no Linux WAV player found. Install/configure alsa-utils for aplay.',file=sys.stderr)
    return 1 if failed else 0


if __name__=='__main__':
    sys.exit(main())
