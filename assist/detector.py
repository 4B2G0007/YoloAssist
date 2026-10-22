from pathlib import Path

class Detector:
    def __init__(self, config):
        import torch
        from ultralytics import YOLO
        model_path = Path(config['model'])
        if not model_path.is_file():
            raise FileNotFoundError(f'Model not found: {model_path}. Copy your yolo11n.pt here or use --model PATH.')
        self.c = config
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if config['device']=='auto' else config['device']
        self.model = YOLO(str(model_path), task='detect')
        names = self.model.names
        from .config import CLASSES
        expected = {k:v[0] for k,v in CLASSES.items()}
        if any(names.get(k) != v for k,v in expected.items()):
            raise ValueError('This program requires a COCO detection model with the expected class IDs.')

    def predict(self, frame):
        result = self.model.predict(source=frame, device=self.device, imgsz=self.c['imgsz'],
                                    conf=self.c['model_conf'], iou=self.c['nms_iou'],
                                    verbose=False, save=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [{'xyxy': b, 'confidence': conf, 'class_id': cls}
                for b,conf,cls in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist())]
