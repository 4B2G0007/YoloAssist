"""斑馬線、上下樓梯、紅黃綠燈（微調模型新增類別）接進警示流程的行為測試。"""
import copy
import unittest
from pathlib import Path

from assist.config import CLASS_RANK, CLASSES, DEFAULT, LIGHT_COLOR_CLASSES, NEW_CLASSES, validate
from assist.engine import Engine
from assist.runner import light_candidates
from assist.voice_catalog import additions

ROOT = Path(__file__).resolve().parents[1]


def det(x1, y1, x2, y2, cls, conf=.9):
    return {'class_id': cls, 'confidence': conf, 'xyxy': [x1*1000, y1*1000, x2*1000, y2*1000]}


class SceneClassTests(unittest.TestCase):
    def setUp(self):
        self.c = copy.deepcopy(DEFAULT)
        self.e = Engine(self.c)

    def test_classes_and_thresholds(self):
        self.assertEqual([CLASSES[k][0] for k in (80, 81, 82)], ['crosswalk', 'stairs_up', 'stairs_down'])
        self.assertTrue({80, 81, 82} <= NEW_CLASSES)
        self.assertTrue(set(LIGHT_COLOR_CLASSES).isdisjoint(CLASSES))  # 燈色不進語音
        validate(copy.deepcopy(DEFAULT))
        self.assertGreater(CLASS_RANK[82], CLASS_RANK[81])
        self.assertGreater(CLASS_RANK[81], CLASS_RANK[2])
        self.assertGreater(CLASS_RANK[2], CLASS_RANK[0])

    def test_voice_files_exist_for_new_classes(self):
        names = [n for n in additions() if any(s in n for s in ('crosswalk', 'stairs_up', 'stairs_down'))]
        self.assertEqual(len(names), 12)
        for n in names:
            self.assertTrue((ROOT / 'audio' / n).is_file(), n)

    def test_stairs_down_immediate_alert(self):
        rows, _, ch, _ = self.e.update([det(.35, .70, .65, 1.0, 82)], 1000, 1000, 0)
        self.assertEqual(rows[0]['state'], 'near')
        self.assertEqual(ch['text'], '注意，前方有下樓梯。')
        self.assertEqual(ch['clip'], 'urgent_center_stairs_down.wav')

    def test_crosswalk_needs_confirmation_then_alerts(self):
        box = det(.1, .80, .9, 1.0, 80)  # 高 0.20：介於 near 0.15 與 immediate 0.35 之間
        _, _, ch, _ = self.e.update([box], 1000, 1000, 0)
        self.assertIsNone(ch)
        _, _, ch, _ = self.e.update([box], 1000, 1000, .2)
        self.assertEqual(ch['text'], '注意，前方有斑馬線。')

    def test_far_crosswalk_is_silent(self):
        box = det(.3, .45, .7, .50, 80)  # 遠處、框很矮
        for t in (0, .2, .4):
            _, _, ch, _ = self.e.update([box], 1000, 1000, t)
            self.assertIsNone(ch)

    def test_stairs_outrank_vehicle_and_person(self):
        car = det(.35, .10, .65, .95, 2)       # 車輛：立即門檻
        person = det(.40, .10, .60, .95, 0)    # 行人：立即門檻
        stairs = det(.30, .72, .70, 1.0, 82)   # 下樓梯：立即門檻
        _, events, ch, _ = self.e.update([person, car, stairs], 1000, 1000, 0)
        self.assertEqual(ch['class_name'], 'stairs_down')
        order = [e['class_name'] for e in events]
        self.assertEqual(order[:3], ['stairs_down', 'car', 'person'])

    def test_light_colors_never_spoken(self):
        for cls in LIGHT_COLOR_CLASSES:
            e = Engine(self.c)
            for t in (0, .2, .4):
                rows, events, ch, _ = e.update([det(.4, .1, .6, .9, cls)], 1000, 1000, t)
                self.assertEqual(rows[0]['reason'], 'non_target_class')
                self.assertEqual(events, [])
                self.assertIsNone(ch)

    def test_light_candidates_remap_for_traffic_light_v2(self):
        cands = [det(0, 0, .1, .1, 83), det(0, 0, .1, .1, 85), det(0, 0, .1, .1, 2), det(0, 0, .1, .1, 9)]
        out = light_candidates(cands, 9)
        self.assertEqual([c['class_id'] for c in out], [9, 9, 2, 9])
        self.assertEqual([c.get('model_color') for c in out], ['red_light', 'green_light', None, None])
        self.assertEqual([c['class_id'] for c in cands], [83, 85, 2, 9])  # 原始資料不被改動


@unittest.skipUnless((ROOT / 'yolo11n.pt').is_file(), 'needs yolo11n.pt')
class DetectorCompatibilityTests(unittest.TestCase):
    def load(self, model):
        from assist.detector import Detector
        c = copy.deepcopy(DEFAULT)
        c['model'], c['device'] = str(model), 'cpu'
        return Detector(c)

    def test_original_coco_model_still_loads(self):
        self.assertEqual(len(self.load(ROOT / 'yolo11n.pt').model.names), 80)

    @unittest.skipUnless((ROOT / 'weights' / 'blindassist_v1.pt').is_file(), 'needs fine-tuned model')
    def test_finetuned_model_loads(self):
        names = self.load(ROOT / 'weights' / 'blindassist_v1.pt').model.names
        self.assertEqual(names[80], 'crosswalk')
        self.assertEqual(names[85], 'green_light')


if __name__ == '__main__':
    unittest.main()
