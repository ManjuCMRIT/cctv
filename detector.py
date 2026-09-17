"""
Core detection + tracking logic for gadget-use monitoring.

- YOLOv8n (pretrained on COCO) detects `person`, `cell phone`, `laptop`.
- A gadget box is attributed to a person if its center falls inside
  (or just outside) that person's box.
- A lightweight IoU-based tracker keeps a stable "Child N" id across
  frames so we can accumulate per-child gadget-use duration.
"""

from dataclasses import dataclass, field

from ultralytics import YOLO

PERSON_CLASS = 0
GADGET_CLASSES = {63: "laptop", 67: "cell phone"}


@dataclass
class TrackedPerson:
    track_id: int
    box: tuple
    gadget_active: bool = False
    gadget_seconds: float = 0.0


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def centroid_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def box_diag(box):
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def point_in_box(point, box, margin=0.0):
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


class GadgetUseTracker:
    """Runs YOLO on frames and keeps per-child gadget-use state over time."""

    def __init__(self, model_path="yolov8n.pt", conf=0.35, match_iou=0.15,
                 match_centroid_ratio=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.match_iou = match_iou
        # A candidate also counts as a match if its box center moved less
        # than (match_centroid_ratio * previous box diagonal) since the last
        # frame - this catches cases where IoU dips (angle/lean/occlusion)
        # but it's clearly still the same person, so their gadget-use timer
        # keeps accumulating instead of resetting on a fresh id.
        self.match_centroid_ratio = match_centroid_ratio
        self.tracks: dict[int, TrackedPerson] = {}
        self._next_id = 1

    def _match_or_create_track(self, box):
        best_id, best_score = None, 0.0
        for tid, tp in self.tracks.items():
            iou_score = iou(tp.box, box)
            dist = centroid_distance(tp.box, box)
            diag = box_diag(tp.box) or 1.0
            centroid_ok = dist <= self.match_centroid_ratio * diag
            if iou_score >= self.match_iou or centroid_ok:
                # prefer the candidate with the best IoU among acceptable matches
                if iou_score > best_score or best_id is None:
                    best_id, best_score = tid, iou_score
        if best_id is not None:
            return best_id
        new_id = self._next_id
        self._next_id += 1
        self.tracks[new_id] = TrackedPerson(track_id=new_id, box=box)
        return new_id

    def process_frame(self, frame, dt):
        """Run detection on one BGR frame. dt = seconds this frame represents."""
        # Run at a lower confidence for gadget classes specifically -- small,
        # partly-occluded phones are the main source of missed detections,
        # and a missed detection here is what makes duration look lower
        # than what you actually see in the video.
        results = self.model.predict(frame, conf=min(self.conf, 0.2), verbose=False)[0]

        person_boxes = []
        gadget_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            xyxy = tuple(box.xyxy[0].tolist())
            if cls_id == PERSON_CLASS and score >= self.conf:
                person_boxes.append(xyxy)
            elif cls_id in GADGET_CLASSES:
                gadget_boxes.append((xyxy, GADGET_CLASSES[cls_id], score))

        self.last_gadget_detections = gadget_boxes  # for debug/inspection

        annotations = []
        for pbox in person_boxes:
            tid = self._match_or_create_track(pbox)
            tp = self.tracks[tid]
            tp.box = pbox

            active, gadget_label = False, None
            for gbox, label, _score in gadget_boxes:
                if point_in_box(box_center(gbox), pbox, margin=20):
                    active, gadget_label = True, label
                    break

            tp.gadget_active = active
            if active:
                tp.gadget_seconds += dt

            annotations.append({
                "track_id": tid,
                "box": pbox,
                "gadget_active": active,
                "gadget_label": gadget_label,
                "gadget_seconds": tp.gadget_seconds,
            })

        return annotations
