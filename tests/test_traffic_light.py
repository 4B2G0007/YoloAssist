"""Synthetic images + mocked detections only; NOT real YOLO/street accuracy."""
import copy
import csv
import io
import json
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import main as entry
from assist.config import CLASSES, DEFAULT
from assist.detector import Detector
from assist.render import Overlay
from assist.runner import run
from assist.traffic_light import (BGR, TrafficLightAnalyzer, TrafficLightOptions,
                                 TemporalFilter, box_iou, classify_light,
                                 draw_lights, resolve_class_id)


def lamp(color='RED', background=25):
    image = np.full((120, 48, 3), background, np.uint8)
    if color != 'UNKNOWN':
        cy = {'RED': 22, 'YELLOW': 60, 'GREEN': 98}[color]
        cv2.circle(image, (24, cy), 10, BGR[color], -1)
    return image


LIGHT_BOX = [20, 90, 68, 210]
OBJECTS = [{'class_id': cid, 'confidence': .92, 'xyxy': rect}
           for cid, rect in [(0, [230, 80, 305, 350]),
                             (2, [325, 85, 485, 350]),
                             (3, [510, 100, 630, 350])]]


def candidate(score=.9, rect=None, class_id=9):
    return {'class_id': class_id, 'confidence': score,
            'xyxy': LIGHT_BOX if rect is None else rect}


def scene(color='RED'):
    gray = np.random.default_rng(7).integers(60, 130, (360, 640), dtype=np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[90:210, 20:68] = lamp(color)
    return image


class AppearanceTests(unittest.TestCase):
    def test_red_yellow_green(self):
        for color in ['RED', 'YELLOW', 'GREEN']:
            with self.subTest(color=color):
                raw, quality = classify_light(lamp(color))
                self.assertEqual(raw, color)
                self.assertGreater(quality, 0)

    def test_uniform_yellow_sign_and_rectangular_band_rejected(self):
        sign = np.full((90, 180, 3), (0, 255, 255), np.uint8)
        band = np.full((100, 100, 3), 25, np.uint8)
        cv2.rectangle(band, (20, 45), (80, 50), (0, 255, 255), -1)
        for image in [sign, band]:
            self.assertEqual(classify_light(image)[0], 'UNKNOWN')

    def test_low_local_contrast_rejected(self):
        self.assertEqual(classify_light(lamp('YELLOW', 240), 22)[0], 'UNKNOWN')
        self.assertEqual(classify_light(lamp('YELLOW', 240), 10)[0], 'YELLOW')

    def test_wrong_vertical_position_rejected(self):
        image = np.full((120, 48, 3), 25, np.uint8)
        cv2.circle(image, (24, 98), 10, BGR['RED'], -1)
        self.assertEqual(classify_light(image)[0], 'UNKNOWN')

    def test_horizontal_order_is_not_assumed(self):
        for color, cx in [('RED', 115), ('GREEN', 25)]:
            image = np.full((48, 144, 3), 25, np.uint8)
            cv2.circle(image, (cx, 24), 10, BGR[color], -1)
            self.assertEqual(classify_light(image)[0], color)

    def test_ambiguous_different_lamps_rejected_even_after_two_same_colors(self):
        for colors in [('RED', 'GREEN'), ('RED', 'RED', 'GREEN')]:
            image = np.full((144, 144, 3), 25, np.uint8)
            for cx, color in zip([26, 70, 114], colors):
                cv2.circle(image, (cx, 70), 9, BGR[color], -1)
            self.assertEqual(classify_light(image)[0], 'UNKNOWN')

    def test_empty_tiny_and_colorless_crops(self):
        for image in [None, np.zeros((0, 0, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), lamp('UNKNOWN')]:
            self.assertEqual(classify_light(image), ('UNKNOWN', 0.0))

    def test_red_hue_wraparound(self):
        image = lamp('RED')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0][hsv[:, :, 1] > 100] = 178
        self.assertEqual(classify_light(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))[0], 'RED')


class TrackingTests(unittest.TestCase):
    rect = (0, 0, 48, 120)

    def test_three_of_five_with_unknown_and_current_evidence_required(self):
        tracker = TemporalFilter(3)
        states = [tracker.update([(self.rect, color)])[0][1]
                  for color in ['RED', 'RED', 'UNKNOWN', 'RED', 'UNKNOWN']]
        self.assertEqual(states, ['UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'RED', 'UNKNOWN'])

    def test_color_change_needs_its_own_votes(self):
        tracker = TemporalFilter(3)
        states = [tracker.update([(self.rect, color)])[0][1]
                  for color in ['RED']*3+['GREEN']*3]
        self.assertEqual(states, ['UNKNOWN', 'UNKNOWN', 'RED', 'UNKNOWN', 'UNKNOWN', 'GREEN'])

    def test_missed_inferences_displace_old_votes(self):
        tracker = TemporalFilter(3)
        for _ in range(3):tracker.update([(self.rect, 'RED')])
        for _ in range(5):tracker.update([])
        result = tracker.update([(self.rect, 'RED')])[0]
        self.assertEqual((result[1], result[2]), ('UNKNOWN', 1))
        for _ in range(9):tracker.update([])
        self.assertFalse(tracker.tracks)

    def test_iou_tracks_remain_independent_after_reordering(self):
        a, b = self.rect, (150, 0, 198, 120)
        tracker = TemporalFilter(3)
        first = tracker.update([(a, 'RED'), (b, 'GREEN')])
        second = tracker.update([(b, 'GREEN'), (a, 'RED')])
        third = tracker.update([(a, 'RED'), (b, 'GREEN')])
        self.assertEqual([x[0] for x in first], [x[0] for x in third])
        self.assertEqual(first[0][0], second[1][0])
        self.assertEqual([x[1] for x in third], ['RED', 'GREEN'])
        fresh = tracker.update([((300, 0, 348, 120), 'RED')])[0]
        self.assertEqual((fresh[1], fresh[2]), ('UNKNOWN', 1))

    def test_reset_gap_and_backward_source_time(self):
        analyzer = TrafficLightAnalyzer(max_gap_s=2)
        for t in [0, .1, .2]:result = analyzer.update(scene(), [candidate()], t)
        self.assertEqual(result[0]['color'], 'RED')
        self.assertEqual(analyzer.update(scene(), [candidate()], 3)[0]['supporting_observations'], 1)
        analyzer.update(scene(), [candidate()], 3.1)
        self.assertEqual(analyzer.update(scene(), [candidate()], 0)[0]['supporting_observations'], 1)
        analyzer.reset()
        self.assertFalse(analyzer.temporal.tracks)

    def test_yolo_gate_and_nontraffic_objects_do_not_vote(self):
        analyzer = TrafficLightAnalyzer()
        items = analyzer.update(scene(), [candidate(.29), candidate(.99, class_id=2)], 0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['reason'], 'below_yolo_confidence')
        self.assertEqual(items[0]['color'], 'UNKNOWN')
        self.assertFalse(analyzer.temporal.tracks)
        self.assertEqual(analyzer.update(scene(), [candidate(.30)], .1)[0]['supporting_observations'], 1)

    def test_screenshot_score_031_is_analyzed_and_confirms_on_second_inference(self):
        analyzer = TrafficLightAnalyzer()
        first = analyzer.update(scene(), [candidate(.31)], 0)[0]
        self.assertEqual((first['raw_color'], first['color'], first['reason']),
                         ('RED', 'UNKNOWN', 'waiting_confirmation'))
        second = analyzer.update(scene(), [candidate(.31)], 1/30)[0]
        self.assertEqual((second['color'], second['supporting_observations']), ('RED', 2))

    def test_legacy_confidence_and_confirmation_can_still_be_requested(self):
        analyzer = TrafficLightAnalyzer(TrafficLightOptions(confidence=.4, min_frames=3))
        rejected = analyzer.update(scene(), [candidate(.31)], 0)[0]
        self.assertEqual(rejected['reason'], 'below_yolo_confidence')
        colors = [analyzer.update(scene(), [candidate(.5)], t/30)[0]['color'] for t in [1, 2, 3]]
        self.assertEqual(colors, ['UNKNOWN', 'UNKNOWN', 'RED'])

    def test_persistent_uniform_yellow_sign_is_unknown(self):
        analyzer = TrafficLightAnalyzer()
        image = scene();image[90:210, 20:68] = (0, 255, 255)
        for index in range(7):items = analyzer.update(image, [candidate()], index*.1)
        self.assertEqual(items[0]['color'], 'UNKNOWN')
        self.assertEqual(items[0]['reason'], 'appearance_rejected')

    def test_invalid_and_outside_boxes_are_safe(self):
        analyzer = TrafficLightAnalyzer()
        items = analyzer.update(scene(), [candidate(rect=[-5, -5, 0, 0]),
                                          candidate(rect=[1, 1, float('nan'), 3])], 0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['reason'], 'invalid_box')
        self.assertEqual(box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0)


class OptionTests(unittest.TestCase):
    def test_invalid_options(self):
        for key, value in [('confidence', -.1), ('confidence', 1.1), ('confidence', float('nan')),
                           ('min_contrast', -1), ('min_contrast', float('inf')), ('min_contrast', 256),
                           ('min_frames', 0), ('min_frames', 6), ('min_frames', True), ('min_frames', 2.5)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                TrafficLightOptions(**{key: value})

    def test_model_class_metadata(self):
        self.assertEqual(resolve_class_id({9: 'traffic light'}), 9)
        self.assertEqual(resolve_class_id({41: 'traffic_light'}), 41)
        self.assertEqual(resolve_class_id(['person', 'traffic light']), 1)
        for names in [{9: 'stop sign'}, {9: 'traffic light', 10: 'traffic light'}]:
            with self.assertRaises(ValueError):resolve_class_id(names)

    def test_cli_defaults_and_existing_parameters(self):
        args = entry.parse_args(['--source', 'videos/street4.mp4', '--show', '--stride', '3',
                                 '--mode', 'classic', '--output', 'out', '--traffic-light'])
        self.assertEqual((args.tl_conf, args.tl_min_contrast, args.tl_min_frames), (.3, 22, 2))
        self.assertFalse(args.audio)
        c, source, _ = entry.configure(args, raspberry_pi=False)
        self.assertEqual((source, c['stride'], c['alert_mode']), ('videos/street4.mp4', 3, 'classic'))

    def test_imgsz_override_and_validation(self):
        normal, _, _ = entry.configure(entry.parse_args(['--source', 'street4.mp4']), raspberry_pi=False)
        self.assertEqual(normal['imgsz'], 640)
        larger, _, _ = entry.configure(entry.parse_args(['--source', 'street4.mp4', '--imgsz', '960']), raspberry_pi=False)
        self.assertEqual(larger['imgsz'], 960)
        with self.assertRaises(ValueError):
            entry.configure(entry.parse_args(['--source', 'street4.mp4', '--imgsz', '0']), raspberry_pi=False)


class Tensor:
    def __init__(self, value):self.value = value
    def cpu(self):return self
    def tolist(self):return self.value


class Model:
    def __init__(self, candidates, names=None):
        self.names = {key: value[0] for key, value in CLASSES.items()} if names is None else names
        if names is None:self.names[9] = 'traffic light'
        self.candidates = candidates
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        # Mock the YOLO confidence filter too, so detector regressions are observable.
        items = [x for x in self.candidates if x['confidence'] >= kwargs['conf']]
        boxes = types.SimpleNamespace(xyxy=Tensor([x['xyxy'] for x in items]),
                                      conf=Tensor([x['confidence'] for x in items]),
                                      cls=Tensor([x['class_id'] for x in items]))
        return [types.SimpleNamespace(boxes=boxes)]


def mock_modules(model, cuda=False):
    constructor = Mock(return_value=model)
    return {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda)),
            'ultralytics': types.SimpleNamespace(YOLO=constructor)}, constructor


class DetectorTests(unittest.TestCase):
    def make_detector(self, model, options=None, cuda=False, **config_changes):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)/'fake.pt';path.touch()
        config = copy.deepcopy(DEFAULT);config.update(model=str(path), **config_changes)
        modules, constructor = mock_modules(model, cuda)
        with patch.dict(sys.modules, modules):detector = Detector(config, options)
        return detector, constructor

    def test_one_model_one_predict_and_same_full_frame(self):
        model = Model(OBJECTS+[candidate()])
        detector, constructor = self.make_detector(model, TrafficLightOptions())
        image = scene();result = detector.predict(image)
        constructor.assert_called_once()
        self.assertEqual(len(model.calls), 1)
        self.assertIs(model.calls[0]['source'], image)
        self.assertNotIn('classes', model.calls[0])
        self.assertEqual({int(x['class_id']) for x in result}, {0, 2, 3, 9})

    def test_lower_light_threshold_does_not_admit_lower_confidence_objects(self):
        model = Model([candidate(.4), candidate(.4, class_id=0), candidate(.7, class_id=2)])
        detector, _ = self.make_detector(model, TrafficLightOptions(), model_conf=.6, alert_conf=.7)
        result = detector.predict(scene())
        self.assertEqual(model.calls[0]['conf'], .3)
        self.assertEqual([x['class_id'] for x in result], [9, 2])

    def test_031_light_uses_one_shared_predict_and_requested_resolution(self):
        model = Model(OBJECTS+[candidate(.31)])
        detector, constructor = self.make_detector(model, TrafficLightOptions(), imgsz=960)
        image = scene()
        result = detector.predict(image)
        constructor.assert_called_once()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]['imgsz'], 960)
        analyzer = TrafficLightAnalyzer()
        self.assertEqual(analyzer.update(image, result, 0)[0]['raw_color'], 'RED')
        self.assertEqual({int(x['class_id']) for x in result}, {0, 2, 3, 9})

    def test_off_mode_and_cpu_cuda_selection(self):
        for cuda, requested, expected in [(False, 'auto', 'cpu'), (True, 'auto', '0'), (True, 'cpu', 'cpu')]:
            model = Model([candidate(.2)])
            detector, _ = self.make_detector(model, cuda=cuda, device=requested)
            self.assertEqual(detector.device, expected)
            detector.predict(scene())
            self.assertEqual(model.calls[0]['conf'], DEFAULT['model_conf'])

    def test_wrong_light_metadata_fails_only_when_enabled(self):
        names = {key: value[0] for key, value in CLASSES.items()};names[9] = 'sign'
        self.make_detector(Model([], names))
        with self.assertRaisesRegex(ValueError, 'traffic light'):
            self.make_detector(Model([], names), TrafficLightOptions())


class RenderTests(unittest.TestCase):
    def test_pending_is_immediately_visible_in_cyan_without_debug(self):
        item = TrafficLightAnalyzer().update(scene(), [candidate()], 0)[0]
        image = np.zeros_like(scene())
        with patch('assist.traffic_light.cv2.putText', wraps=cv2.putText) as put:
            draw_lights(image, [item], False)
            self.assertIn('LIGHT: CHECKING | YOLO 0.90', put.call_args.args[1])
            self.assertEqual(put.call_args.args[5], (255, 255, 0))
            np.testing.assert_array_equal(image[150, 20], BGR['UNKNOWN'])
            self.assertEqual(item['color'], 'UNKNOWN')
            put.reset_mock()
            draw_lights(image, [item], True)
            labels = [call.args[1] for call in put.call_args_list]
            self.assertTrue(any('LIGHT: CHECKING | YOLO 0.90' in x for x in labels))
            self.assertTrue(any('raw=RED votes=1/2' in x for x in labels))

    def test_rejected_candidates_are_debug_only_and_not_signal_colors(self):
        item = TrafficLightAnalyzer().update(scene(), [candidate(.25)], 0)[0]
        original = np.zeros_like(scene())
        image = original.copy()
        draw_lights(image, [item])
        np.testing.assert_array_equal(image, original)
        with patch('assist.traffic_light.cv2.putText', wraps=cv2.putText) as put:
            draw_lights(image, [item], True)
        self.assertTrue(any('LIGHT: LOW CONF' in c.args[1] for c in put.call_args_list))
        np.testing.assert_array_equal(image[150, 20], BGR['REJECTED'])

    def test_unknown_appearance_is_visible_but_never_colored_as_a_signal(self):
        image = scene(); image[90:210, 20:68] = (0, 255, 255)
        analyzer = TrafficLightAnalyzer()
        for t in range(7):
            item = analyzer.update(image, [candidate(.31)], t/30)[0]
        with patch('assist.traffic_light.cv2.putText', wraps=cv2.putText) as put:
            draw_lights(image, [item])
        self.assertIn('LIGHT: UNKNOWN', put.call_args.args[1])
        self.assertEqual(put.call_args.args[5], BGR['UNKNOWN'])
        self.assertEqual(item['color'], 'UNKNOWN')

    def test_labels_do_not_cover_lamp_interior_and_boxes_keep_yolo_coordinates(self):
        original = scene()
        item = TrafficLightAnalyzer().update(original, [candidate(.31)], 0)[0]
        for debug in [False, True]:
            image = original.copy()
            draw_lights(image, [item], debug)
            np.testing.assert_array_equal(image[100:200, 29:59], original[100:200, 29:59])
            np.testing.assert_array_equal(image[150, 20], BGR['UNKNOWN'])
            np.testing.assert_array_equal(image[150, 68], BGR['UNKNOWN'])
            self.assertEqual(item['xyxy'], LIGHT_BOX)
            self.assertEqual(item['supporting_observations'], 1)

    def test_upper_image_signal_is_visible_over_the_original_header(self):
        image = scene()
        image[5:125, 450:498] = lamp()
        item = TrafficLightAnalyzer().update(image, [candidate(.31, [450, 5, 498, 125])], 0)[0]
        overlay = Overlay(copy.deepcopy(DEFAULT))
        result = overlay.draw(image, [], 'original title', 'original prompt', traffic_lights=[item])
        np.testing.assert_array_equal(result[15, 450], BGR['UNKNOWN'])

    def test_tiny_or_edge_touching_frames_do_not_raise(self):
        analyzer = TrafficLightAnalyzer()
        for h, w in [(8, 8), (30, 30), (95, 531), (360, 640)]:
            image = np.zeros((h, w, 3), np.uint8)
            item = analyzer.update(image, [candidate(.31, [0, 0, w, h])], 0)[0]
            self.assertIs(draw_lights(image, [item], True), image)

    def test_confirmed_colors_and_held_label(self):
        for color in ['RED', 'YELLOW', 'GREEN']:
            analyzer = TrafficLightAnalyzer(TrafficLightOptions(min_frames=1))
            item = analyzer.update(scene(color), [candidate()], 0)[0]
            with patch('assist.traffic_light.cv2.putText', wraps=cv2.putText) as put:
                draw_lights(scene(color), [item], age=.1)
            self.assertIn(f'LIGHT: {color}', put.call_args.args[1])
            self.assertIn('HELD 0.10s', put.call_args.args[1])
            self.assertEqual(put.call_args.args[5], BGR[color])

    def test_original_header_and_object_overlay_remain(self):
        from assist.engine import Engine
        config = copy.deepcopy(DEFAULT)
        rows, _, _, _ = Engine(config).update(OBJECTS, 640, 360, 0)
        overlay = Overlay(config)
        analyzer = TrafficLightAnalyzer(TrafficLightOptions(min_frames=1))
        items = analyzer.update(scene(), [candidate()], 0)
        before = overlay.draw(scene(), rows, 'original title', 'original prompt')
        after = overlay.draw(scene(), rows, 'original title', 'original prompt', traffic_lights=items)
        np.testing.assert_array_equal(before[:80], after[:80])
        np.testing.assert_array_equal(before[180:355, 220:], after[180:355, 220:])
        self.assertFalse(np.array_equal(before, after))


class FakeDetector:
    device = 'fake_NOT_real_YOLO'
    traffic_light_class_id = 9
    def __init__(self):self.calls = 0
    def predict(self, frame):
        self.calls += 1
        return copy.deepcopy(OBJECTS+[candidate()])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory();cls.root = Path(cls.temporary.name)
        cls.video = cls.root/'street4_synthetic.mp4'
        cls.colors = cls.root/'colors_synthetic.mp4'
        for path, colors in [(cls.video, ['RED']*21),
                             (cls.colors, ['RED']*7+['YELLOW']*7+['GREEN']*7)]:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 360))
            if not writer.isOpened():raise RuntimeError('Synthetic MP4 writer unavailable')
            for color in colors:writer.write(scene(color))
            writer.release()

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def run_video(self, options=None, path=None, detector=None, **kwargs):
        config = copy.deepcopy(DEFAULT);config.update(kwargs.pop('config', {}))
        return run(path or self.video, config, detector or FakeDetector(),
                   output=self.root/'results', traffic_light=options, **kwargs)

    def logs(self, report):
        folder = Path(report['output_directory'])
        return [json.loads(line) for line in next(folder.glob('*_detections.jsonl')).read_text().splitlines()]

    def test_on_and_off_preserve_objects_directions_danger_and_csv(self):
        off = self.run_video();on = self.run_video(TrafficLightOptions())
        left, right = self.logs(off), self.logs(on)
        self.assertNotIn('traffic_lights', left[0])
        for a, b in zip(left, right):
            self.assertEqual(a['objects'], b['objects'])
            aa = [{k: v for k, v in row.items() if k != 'monotonic_s'} for row in a['decisions']]
            bb = [{k: v for k, v in row.items() if k != 'monotonic_s'} for row in b['decisions']]
            self.assertEqual(aa, bb)
        eligible = [row for row in right[2]['objects'] if row['eligible']]
        self.assertEqual({x['class_name'] for x in eligible}, {'person', 'car', 'motorcycle'})
        self.assertTrue(all(x['direction'] and x['state'] for x in eligible))
        self.assertTrue(any(x['state'] == 'near' for x in eligible))
        self.assertEqual((on['video']['written_frames'], on['video']['verified_frames']), (21, 21))
        for report in [off, on]:
            with next(Path(report['output_directory']).glob('*_events.csv')).open(encoding='utf-8-sig') as f:
                reader = csv.DictReader(f);events = list(reader)
                from assist.recording import FIELDS
                self.assertEqual(reader.fieldnames, FIELDS)
                self.assertFalse(any(x['class_name'] == 'traffic light' for x in events))
            self.assertFalse(report['audio_status_counts'].get('started'))

    def test_all_three_colors_through_encoded_video_and_json(self):
        report = self.run_video(TrafficLightOptions(), path=self.colors)
        logs = self.logs(report)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(logs[0]['traffic_lights'][0]['color'], 'UNKNOWN')
        self.assertEqual(logs[1]['traffic_lights'][0]['color'], 'RED')
        self.assertEqual(logs[7]['traffic_lights'][0]['color'], 'UNKNOWN')
        self.assertEqual(logs[8]['traffic_lights'][0]['color'], 'YELLOW')
        self.assertEqual(logs[14]['traffic_lights'][0]['color'], 'UNKNOWN')
        self.assertEqual(logs[15]['traffic_lights'][0]['color'], 'GREEN')

    def test_stride_counts_inferences_not_reused_frames(self):
        detector = FakeDetector()
        report = self.run_video(TrafficLightOptions(), detector=detector, config={'stride': 3})
        logs = self.logs(report)
        self.assertEqual((report['frames_inferred'], detector.calls), (7, 7))
        self.assertEqual(logs[0]['traffic_lights'][0]['color'], 'UNKNOWN')
        self.assertEqual(logs[3]['traffic_lights'][0]['supporting_observations'], 2)
        self.assertEqual(logs[3]['traffic_lights'][0]['color'], 'RED')
        self.assertEqual(logs[6]['traffic_lights'][0]['supporting_observations'], 3)
        self.assertTrue(all(row['traffic_lights'] == [] for row in logs if not row['inferred']))
        self.assertEqual(report['video']['verified_frames'], 21)

    def test_per_source_state_reset_while_reusing_detector(self):
        detector = FakeDetector()
        for _ in range(2):
            report = self.run_video(TrafficLightOptions(), detector=detector)
            items = self.logs(report)[0]['traffic_lights']
            self.assertEqual((items[0]['color'], items[0]['track_id'], items[0]['supporting_observations']),
                             ('UNKNOWN', 1, 1))

    def test_show_uses_one_existing_window_with_combined_frame(self):
        with patch('cv2.imshow') as show, patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            report = self.run_video(TrafficLightOptions(), display=True, record=False)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(show.call_count, 21)
        self.assertEqual({x.args[0] for x in show.call_args_list}, {'YOLO Assist V1'})
        self.assertEqual(show.call_args.args[1].shape, (360, 640, 3))

    def test_no_record_max_frames_and_debug(self):
        report = self.run_video(TrafficLightOptions(debug=True), record=False, max_frames=2)
        self.assertEqual(report['status'], 'limited')
        self.assertIsNone(report['output_directory'])

    def test_long_stride_expires_cached_light_and_old_votes(self):
        # At 10 FPS, a 6-frame gap exceeds this configured 0.4 second TTL.
        report = self.run_video(TrafficLightOptions(), config={'stride': 6, 'stale_seconds': .4})
        logs = self.logs(report)
        inferred = [row for row in logs if row['inferred']]
        self.assertTrue(all(row['traffic_lights'][0]['supporting_observations'] == 1 for row in inferred))
        self.assertTrue(all(row['traffic_lights'][0]['color'] == 'UNKNOWN' for row in inferred))

    def test_stale_camera_suppresses_light_and_resets_votes(self):
        class Camera:
            def __init__(self, *args):pass
            def read(self, last, timeout):return last+1, scene(), time.monotonic()-5
            def close(self):pass
        with patch('assist.runner.LatestCamera', Camera):
            report = run('0', copy.deepcopy(DEFAULT), FakeDetector(), output=self.root/'results',
                         max_frames=4, traffic_light=TrafficLightOptions())
        logs = self.logs(report)
        self.assertTrue(all(x['stale_result'] and x['traffic_lights'] == [] for x in logs))
        self.assertEqual(report['traffic_light_observations'], {})

    def test_main_batch_uses_one_model_and_resets_each_video(self):
        batch = self.root/'batch';batch.mkdir(exist_ok=True)
        for name in ['street10.mp4', 'street2.mp4']:
            (batch/name).write_bytes(self.video.read_bytes())
        model = Model(OBJECTS+[candidate()]);modules, constructor = mock_modules(model)
        weights = self.root/'fake.pt';weights.touch()
        out = self.root/'batch_results'
        argv = ['main.py', '--source', str(batch), '--model', str(weights), '--output', str(out),
                '--traffic-light', '--stride', '3', '--tl-debug', '--mode', 'classic']
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(), 0)
        constructor.assert_called_once()
        self.assertEqual(len(model.calls), 14)
        summary = json.loads(next(out.glob('batch_*.json')).read_text())
        self.assertEqual([Path(x['source']).name for x in summary['results']], ['street2.mp4', 'street10.mp4'])
        for report in summary['results']:
            first = self.logs(report)[0]['traffic_lights'][0]
            self.assertEqual((first['color'], first['supporting_observations']), ('UNKNOWN', 1))

    def test_main_off_without_audio_or_recording_still_runs(self):
        model = Model(OBJECTS+[candidate()]);modules, constructor = mock_modules(model)
        weights = self.root/'fake_off.pt';weights.touch()
        argv = ['main.py', '--source', str(self.video), '--model', str(weights), '--no-record']
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(), 0)
        constructor.assert_called_once()
        self.assertEqual(len(model.calls), 21)
        self.assertTrue(all(x['conf'] == DEFAULT['model_conf'] for x in model.calls))


if __name__ == '__main__':
    unittest.main()
