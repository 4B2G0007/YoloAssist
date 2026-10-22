"""Camera worker stores only its newest frame. File reads remain sequential."""
import threading
import time
import sys


def picamera_frame_to_bgr(frame, pixel_format):
    """Convert Picamera2's in-memory channel order, not its counterintuitive name.

    XBGR8888=[R,G,B,X], XRGB8888=[B,G,R,X], RGB888=[B,G,R],
    BGR888=[R,G,B]. YOLO/OpenCV numpy input must be uint8 BGR.
    """
    import cv2
    import numpy as np
    channels = {'XBGR8888':4, 'XRGB8888':4, 'RGB888':3, 'BGR888':3}
    if pixel_format not in channels:
        raise ValueError(f'Unsupported Picamera2 format: {pixel_format}')
    if (frame is None or getattr(frame,'ndim',0)!=3 or
            frame.shape[2]!=channels[pixel_format] or frame.dtype!=np.uint8):
        raise ValueError(f'Expected uint8 HxWx{channels[pixel_format]} frame for {pixel_format}')
    if pixel_format=='XBGR8888':
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if pixel_format=='XRGB8888':
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if pixel_format=='BGR888':
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)

class LatestCamera:
    def __init__(self, source, size, options=None):
        self.source, self.size = source, size
        self.options = dict(options or {})
        self.info = None
        self.cv = threading.Condition()
        self.latest, self.serial = None, 0
        self.error, self.stop_event = None, threading.Event()
        self.cap = self.picam = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        try:
            if self.source == 'picamera2':
                try:
                    from picamera2 import Picamera2
                except ImportError as exc:
                    raise RuntimeError('Picamera2/libcamera unavailable in this Python environment. '
                                       'Use Raspberry Pi OS python3, apt package python3-picamera2, '
                                       'and a venv created with --system-site-packages. '
                                       f'Original error: {exc}') from exc
                pixel_format = self.options.get('camera_format','RGB888')
                camera_index = self.options.get('camera_index',0)
                fps = self.options.get('camera_fps',30.0)
                autofocus = self.options.get('camera_autofocus','continuous')
                self.picam = Picamera2(camera_num=camera_index)
                self.picam.configure(self.picam.create_video_configuration(
                    main={'size': tuple(self.size), 'format': pixel_format},
                    controls={'FrameRate':fps}, buffer_count=2, queue=False))
                self.info = {'camera_index':camera_index, 'requested_size':list(self.size),
                             'pixel_format':pixel_format, 'output_color_order':'BGR',
                             'requested_capture_fps':fps, 'autofocus_requested':autofocus,
                             'autofocus_applied':False, 'warnings':[]}
                if autofocus=='continuous':
                    if 'AfMode' in self.picam.camera_controls:
                        from libcamera import controls
                        self.picam.set_controls({'AfMode':controls.AfModeEnum.Continuous})
                        self.info['autofocus_applied']=True
                    else:
                        warning='Camera does not advertise autofocus; continuing without AfMode.'
                        self.info['warnings'].append(warning)
                        print('[Camera warning] '+warning,file=sys.stderr)
                self.picam.start()
                # Do not expose startup exposure/focus settling frames to the rules.
                if self.stop_event.wait(self.options.get('camera_warmup_s',1.0)):
                    return
                grab = lambda: (True, self.picam.capture_array('main'))
            else:
                import cv2
                self.cap = cv2.VideoCapture(int(self.source))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    raise RuntimeError(f'Cannot open camera {self.source}')
                grab = self.cap.read
            while not self.stop_event.is_set():
                ok, frame = grab()
                acquired = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError('Camera stream ended / image unavailable')
                if self.source=='picamera2':
                    frame = picamera_frame_to_bgr(frame,pixel_format)
                with self.cv:
                    self.serial += 1
                    self.latest = (self.serial, frame, acquired)
                    self.cv.notify_all()
        except Exception as e:
            with self.cv:
                hint = (' Stop stream.py, rpicam-hello and any other camera process before retrying.'
                        if self.source=='picamera2' else '')
                self.error = f'{type(e).__name__}: {e}{hint}'
                self.cv.notify_all()
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.picam is not None:
                try:
                    self.picam.stop()
                except Exception:
                    pass
                try:
                    self.picam.close()
                except Exception:
                    pass

    def read(self, last_serial, timeout):
        with self.cv:
            self.cv.wait_for(lambda: self.error or self.stop_event.is_set() or
                             (self.latest and self.serial > last_serial), timeout)
            if self.error:
                raise RuntimeError(self.error)
            if self.stop_event.is_set():
                raise RuntimeError('Camera has been stopped')
            if not self.latest or self.serial <= last_serial:
                raise TimeoutError('Camera frame timeout')
            return self.latest

    def close(self):
        self.stop_event.set()
        with self.cv:
            self.cv.notify_all()
        self.thread.join(timeout=2)
