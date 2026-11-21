from pathlib import Path

class Detector:
    def __init__(self, config, traffic_light=None):
        import torch
        from ultralytics import YOLO
        model_path = Path(config['model'])
        if not model_path.exists():  # .pt 是檔案；NCNN 匯出的是資料夾（*_ncnn_model）
            raise FileNotFoundError(f'Model not found: {model_path}. Copy your yolo11n.pt here or use --model PATH.')
        self.c = config
        self.traffic_light = traffic_light
        self.traffic_light_class_id = None
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if config['device']=='auto' else config['device']
        self.model = YOLO(str(model_path), task='detect')
        names = self.model.names
        if not isinstance(names, dict):
            names = dict(enumerate(names))
        from .config import CLASSES
        expected = {k:v[0] for k,v in CLASSES.items()}
        # COCO 類別 id 必須一致；新增類別（80 起）可以沒有（原始 yolo11n.pt），但有的話名稱必須對。
        if any(names.get(k) != v for k,v in expected.items() if k < 80 or k in names):
            raise ValueError('This program requires a COCO detection model with the expected class IDs.')
        self.predict_conf = config['model_conf']
        if traffic_light is not None:
            from .traffic_light import resolve_class_id
            self.traffic_light_class_id = resolve_class_id(names)
            self.predict_conf = min(self.predict_conf, traffic_light.confidence)

    def predict(self, frame):
        result = self.model.predict(source=frame, device=self.device, imgsz=self.c['imgsz'],
                                    conf=self.predict_conf, iou=self.c['nms_iou'],
                                    verbose=False, save=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [{'xyxy': b, 'confidence': conf, 'class_id': cls}
                for b,conf,cls in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist())
                if conf >= self.c['model_conf'] or
                (self.traffic_light is not None and cls == self.traffic_light_class_id
                 and conf >= self.traffic_light.confidence)]
