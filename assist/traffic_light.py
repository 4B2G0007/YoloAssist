"""Traffic Light V2 appearance analysis and inference-tick tracking.

Input is the existing detector's candidates and an unannotated BGR frame.
This module never loads a model, opens a window, or creates audio decisions.
Color quality is an uncalibrated shape/contrast score, not YOLO confidence.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# OpenCV HSV values: H 0..179; S,V 0..255. Conservative starting points.
HSV_RANGES = {
    "RED": (((0, 115, 165), (11, 255, 255)), ((169, 115, 165), (179, 255, 255))),
    "YELLOW": (((14, 115, 165), (36, 255, 255)),),
    "GREEN": (((37, 100, 150), (95, 255, 255)),),
}
BGR = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "UNKNOWN": (255, 255, 0),  # Bright cyan means color unconfirmed, not green.
    "REJECTED": (255, 80, 255),
}


def classify_light(crop: np.ndarray, min_contrast: float = 22.0) -> tuple[str, float]:
    """Return (color, quality) for a bright circular lamp on a darker housing.

    A large uniform patch of color, such as a yellow sign, is intentionally
    rejected. A decorative circle on a dark sign may still look like a lamp.
    """
    if crop is None or crop.size == 0:
        return "UNKNOWN", 0.0
    h, w = crop.shape[:2]
    if h < 12 or w < 10:
        return "UNKNOWN", 0.0

    mx, my = max(1, round(w * 0.03)), max(1, round(h * 0.03))
    img = crop[my:h-my, mx:w-mx]
    if img.size == 0:
        return "UNKNOWN", 0.0

    hh, ww = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    area = hh * ww
    minimum_area = max(5, round(area * 0.003))
    maximum_area = area * 0.30
    candidates: list[tuple[float, str]] = []

    for name, ranges in HSV_RANGES.items():
        mask = np.zeros((hh, ww), dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        if min(hh, ww) >= 28:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, n):
            xx, yy, bw, bh, pixels = [int(v) for v in stats[k]]
            if not (minimum_area <= pixels <= maximum_area) or bw < 3 or bh < 3:
                continue
            aspect = bw / bh
            if not (0.52 <= aspect <= 1.92):
                continue

            component = np.uint8(labels[yy:yy+bh, xx:xx+bw] == k) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            roundness = (4 * np.pi * cv2.contourArea(contour) / (perimeter**2)) if perimeter else 0
            if roundness < 0.43:  # thin letters and rectangular bands are unlikely lamps
                continue

            # The lamp should be brighter than a local ring of surrounding pixels.
            # Uniform yellow sign panels have little brightness contrast.
            component_all = np.uint8(labels == k) * 255
            diameter = max(bw, bh)
            kernel_size = max(5, int(diameter * 0.7) | 1)
            halo = cv2.dilate(component_all,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (kernel_size, kernel_size)))
            ring = cv2.bitwise_and(halo, cv2.bitwise_not(component_all))
            if cv2.countNonZero(ring) < 5:
                continue
            inside_v = float(np.median(value[component_all > 0]))
            outside_v = float(np.median(value[ring > 0]))
            contrast = inside_v - outside_v
            if contrast < min_contrast:
                continue

            # A loose positional check only when the YOLO box clearly includes
            # a full elongated 3-lamp housing. Vertical: R top, Y middle, G bottom.
            cx, cy = centroids[k][0] / ww, centroids[k][1] / hh
            if hh / ww >= 2.1:
                if name == "RED" and cy > 0.52:
                    continue
                if name == "GREEN" and cy < 0.48:
                    continue
                if name == "YELLOW" and not (0.18 <= cy <= 0.82):
                    continue
            elif ww / hh >= 2.1:
                # Lamp order varies with intersection design, so do NOT enforce
                # a red-to-green left/right order on horizontal assemblies.
                pass

            quality = (min(contrast, 120.0) / 120.0) * min(roundness, 1.0) * np.sqrt(pixels)
            candidates.append((quality, name))

    if not candidates:
        return "UNKNOWN", 0.0
    candidates.sort(reverse=True)
    best_quality, best_name = candidates[0]
    # Two different bright lights in the same crop => ambiguous, stay silent.
    if any(name != best_name and score >= 0.75 * best_quality
           for score, name in candidates[1:]):
        return "UNKNOWN", 0.0
    return best_name, float(best_quality)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / (aa + bb - intersection) if (aa + bb - intersection) else 0.0


@dataclass(frozen=True)
class TrafficLightOptions:
    confidence: float = 0.30
    min_contrast: float = 22.0
    min_frames: int = 2
    debug: bool = False

    def __post_init__(self):
        for name, low, high in [('confidence', 0.0, 1.0), ('min_contrast', 0.0, 255.0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'traffic light {name}: must be finite and within {low}..{high}')
        if type(self.min_frames) is not int or not 1 <= self.min_frames <= 5:
            raise ValueError('--tl-min-frames must be from 1 to 5')

    def snapshot(self):
        return dict(asdict(self), history_window=5, time_basis='actual_inference_ticks',
                    color_quality='uncalibrated_shape_contrast_score_not_probability')


def resolve_class_id(names):
    """Resolve by model metadata; do not assume every model uses COCO ID 9."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(key) for key, name in items
               if str(name).strip().lower().replace('_', ' ') == 'traffic light']
    if len(matches) != 1:
        raise ValueError('Traffic-light mode requires exactly one model class named "traffic light"; check your model names.')
    return matches[0]


@dataclass
class Track:
    rect: tuple[int, int, int, int]
    history: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    misses: int = 0


class TemporalFilter:
    """One update is one YOLO inference, including inferences with no boxes.

    Misses and appearance rejection count as UNKNOWN observations. Reused
    preview frames must never call update. A current UNKNOWN is never confirmed
    by old votes, and a new color cannot inherit another color's confirmation.
    """
    def __init__(self, min_frames=2):
        if type(min_frames) is not int or not 1 <= min_frames <= 5:
            raise ValueError('min_frames must be from 1 to 5')
        self.min_frames = min_frames
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        # Global greedy one-to-one IoU assignment, independent of input ordering.
        pairs = [(box_iou(rect, track.rect), index, tid)
                 for index, (rect, _) in enumerate(detections)
                 for tid, track in self.tracks.items()]
        assigned, used = {}, set()
        for score, index, tid in sorted(pairs, reverse=True):
            if score >= 0.12 and index not in assigned and tid not in used:
                assigned[index] = tid
                used.add(tid)
        results = []
        for index, (rect, raw_color) in enumerate(detections):
            tid = assigned.get(index)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(rect)
            track = self.tracks[tid]
            track.rect, track.misses = rect, 0
            track.history.append(raw_color)
            used.add(tid)
            votes = Counter(track.history)[raw_color] if raw_color != 'UNKNOWN' else 0
            confirmed = raw_color if raw_color != 'UNKNOWN' and votes >= self.min_frames else 'UNKNOWN'
            results.append((tid, confirmed, votes, len(track.history)))
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid].misses += 1
                self.tracks[tid].history.append('UNKNOWN')
                if self.tracks[tid].misses > 8:
                    del self.tracks[tid]
        return results


class TrafficLightAnalyzer:
    def __init__(self, options=None, class_id=9, max_gap_s=2.0):
        self.options = options if options is not None else TrafficLightOptions()
        self.class_id = int(class_id)
        self.max_gap_s = max_gap_s
        self.temporal = TemporalFilter(self.options.min_frames)
        self.last_time = None

    def reset(self):
        self.temporal.reset()
        self.last_time = None

    def expire(self, now):
        if self.last_time is not None and (now < self.last_time or now - self.last_time > self.max_gap_s):
            self.reset()

    def update(self, frame, candidates, now):
        self.expire(now)
        self.last_time = now
        h, w = frame.shape[:2]
        items, eligible = [], []
        for candidate in candidates:
            if int(candidate['class_id']) != self.class_id:
                continue
            score = float(candidate['confidence'])
            raw_box = candidate['xyxy']
            if len(raw_box) != 4 or not all(math.isfinite(float(v)) for v in raw_box) or not math.isfinite(score):
                continue
            x1, y1, x2, y2 = (int(v) for v in raw_box)
            rect = (max(0, min(w, x1)), max(0, min(h, y1)),
                    max(0, min(w, x2)), max(0, min(h, y2)))
            item = dict(class_id=self.class_id, xyxy=list(rect), track_id=None,
                        detector_confidence=score, raw_color='UNKNOWN', color='UNKNOWN',
                        color_quality=0.0, supporting_observations=0, window_observations=0,
                        required_observations=self.options.min_frames,
                        minimum_detector_confidence=self.options.confidence,
                        reason='below_yolo_confidence')
            items.append(item)
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                item['reason'] = 'invalid_box'
                continue
            if score < self.options.confidence:
                continue
            raw, quality = classify_light(frame[rect[1]:rect[3], rect[0]:rect[2]], self.options.min_contrast)
            item.update(raw_color=raw, color_quality=quality,
                        reason='appearance_rejected' if raw == 'UNKNOWN' else 'waiting_confirmation')
            eligible.append((item, rect, raw))
        matches = self.temporal.update([(rect, raw) for _, rect, raw in eligible])
        for (item, _, _), (tid, stable, votes, window) in zip(eligible, matches):
            item.update(track_id=tid, color=stable, supporting_observations=votes,
                        window_observations=window)
            if stable != 'UNKNOWN':
                item['reason'] = 'confirmed'
        return items


def _label_rect(box, width, height, frame_size, obstacles):
    """Prefer labels outside lamp/object boxes, below the original header."""
    frame_h, frame_w = frame_size
    x1, y1, x2, y2 = box
    gap = 9
    min_y = min(104, max(0, frame_h-height-5))
    positions = [(x1, y1-gap-height), (x2+gap, y1),
                 (x1-gap-width, y1), (x1, y2+gap)]
    choices = []
    for rank, (x, y) in enumerate(positions):
        x = max(0, min(x, frame_w-width-1))
        y = max(min_y, min(y, frame_h-height-5))
        rect = (x, y, x+width, y+height)
        overlap = sum(max(0, min(rect[2], b[2])-max(x, b[0])) *
                      max(0, min(rect[3], b[3])-max(y, b[1])) for b in obstacles)
        choices.append((overlap, rank, rect))
    return min(choices)[2]


def draw_lights(frame, items, debug=False, age=None, avoid_rects=()):
    """Immediate cyan candidate boxes; signal colors only after confirmation.

    Coordinates always come from the shared detector. Label placement never
    moves or enlarges a detection, and rendering never creates votes.
    """
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return frame
    visible = [item for item in items
               if item['reason'] != 'invalid_box' and
               (debug or item['reason'] != 'below_yolo_confidence') and
               item['xyxy'][2] > item['xyxy'][0] and item['xyxy'][3] > item['xyxy'][1]]
    obstacles = [item['xyxy'] for item in visible] + list(avoid_rects)
    thickness = max(3, min(6, round(min(h, w)/300)))
    # Draw all rectangles first, then place labels without covering lamp boxes.
    for item in visible:
        x1, y1, x2, y2 = item['xyxy']
        rejected = item['reason'] == 'below_yolo_confidence'
        color = BGR['REJECTED'] if rejected else BGR[item['color']]
        corners = ((x1, y1), (min(w-1, x2), min(h-1, y2)))
        cv2.rectangle(frame, *corners, (8, 8, 8), thickness+4)
        cv2.rectangle(frame, *corners, color, thickness)
    for item in visible:
        stable = item['color']
        rejected = item['reason'] == 'below_yolo_confidence'
        color = BGR['REJECTED'] if rejected else BGR[stable]
        state = ('LOW CONF' if rejected else
                 ('CHECKING' if stable == 'UNKNOWN' and item['raw_color'] != 'UNKNOWN' else stable))
        title = f"LIGHT: {state} | YOLO {item['detector_confidence']:.2f}"
        if age is not None:
            title += f' HELD {age:.2f}s'
        labels = [title]
        if debug:
            required = item.get('required_observations', 2)
            labels += [f"raw={item['raw_color']} votes={item['supporting_observations']}/{required} "
                       f"q={item['color_quality']:.1f}", item['reason']]
        scale = 0.54
        font = cv2.FONT_HERSHEY_SIMPLEX
        widest = max(cv2.getTextSize(label, font, scale, 1)[0][0] for label in labels)
        if widest > w-14:
            scale *= (w-14)/widest
        line_height = max(14, round(21*scale/0.54))
        width = min(w-2, max(cv2.getTextSize(label, font, scale, 1)[0][0] for label in labels)+12)
        height = line_height*len(labels)+6
        rect = _label_rect(item['xyxy'], width, height, (h, w), obstacles)
        x, y, right, bottom = rect
        cv2.rectangle(frame, (x, y), (right, bottom), (16, 16, 16), -1)
        cv2.rectangle(frame, (x, y), (right, bottom), color, 2)
        for index, label in enumerate(labels):
            baseline = y+line_height*(index+1)
            cv2.putText(frame, label, (x+6, baseline), font, scale, color, 1, cv2.LINE_AA)
        obstacles.append(rect)
    return frame

