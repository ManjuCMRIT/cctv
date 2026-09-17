from ultralytics import YOLO
import math


class ReadingDetector:

    def __init__(self, model_path="yolov8n-pose.pt", conf=0.30):
        self.model = YOLO(model_path)
        self.conf = conf

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def is_reading(self, keypoints):

        # COCO pose keypoints
        # 0  = nose
        # 5  = left shoulder
        # 6  = right shoulder
        # 7  = left elbow
        # 8  = right elbow
        # 9  = left wrist
        # 10 = right wrist
        # 11 = left hip
        # 12 = right hip

        required = [0, 5, 6, 9, 10, 11, 12]

        # Check keypoint confidence
        for idx in required:
            if keypoints[idx][2] < 0.25:
                return False

        nose = keypoints[0][:2]

        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]

        left_wrist = keypoints[9][:2]
        right_wrist = keypoints[10][:2]

        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        # -----------------------------------------
        # Shoulder and hip reference points
        # -----------------------------------------

        shoulder_x = (
            left_shoulder[0] +
            right_shoulder[0]
        ) / 2

        shoulder_y = (
            left_shoulder[1] +
            right_shoulder[1]
        ) / 2

        hip_x = (
            left_hip[0] +
            right_hip[0]
        ) / 2

        hip_y = (
            left_hip[1] +
            right_hip[1]
        ) / 2

        body_height = abs(hip_y - shoulder_y)

        if body_height < 30:
            return False

        # -----------------------------------------
        # 1. Head position
        # -----------------------------------------

        # How far the nose is below/above shoulders
        nose_vertical_ratio = (
            nose[1] - shoulder_y
        ) / body_height

        # A reading posture normally brings the
        # head closer toward the chest.
        #
        # We use a MUCH more forgiving threshold
        # than the previous version.

        head_lowered = nose_vertical_ratio > -0.45

        # -----------------------------------------
        # 2. Nose should be roughly centered
        # -----------------------------------------

        shoulder_width = abs(
            right_shoulder[0] -
            left_shoulder[0]
        )

        if shoulder_width < 20:
            return False

        horizontal_head_offset = abs(
            nose[0] - shoulder_x
        )

        head_centered = (
            horizontal_head_offset <
            shoulder_width * 0.8
        )

        # -----------------------------------------
        # 3. Hands should be below shoulders
        # -----------------------------------------

        left_hand_down = (
            left_wrist[1] >
            shoulder_y
        )

        right_hand_down = (
            right_wrist[1] >
            shoulder_y
        )

        hands_down = (
            left_hand_down or
            right_hand_down
        )

        # -----------------------------------------
        # 4. Hands should not be extremely far away
        # -----------------------------------------

        left_hand_distance = self.distance(
            left_wrist,
            (shoulder_x, shoulder_y)
        )

        right_hand_distance = self.distance(
            right_wrist,
            (shoulder_x, shoulder_y)
        )

        hands_reasonably_close = (
            left_hand_distance < body_height * 1.5
            or
            right_hand_distance < body_height * 1.5
        )

        # -----------------------------------------
        # Final decision
        # -----------------------------------------

        score = 0

        if head_lowered:
            score += 1

        if head_centered:
            score += 1

        if hands_down:
            score += 1

        if hands_reasonably_close:
            score += 1

        # Need 3 out of 4 conditions
        return score >= 3

    def process_frame(self, frame):

        results = self.model.predict(
            frame,
            conf=self.conf,
            verbose=False
        )[0]

        detections = []

        if results.keypoints is None:
            return detections

        boxes = results.boxes
        keypoints = results.keypoints

        for i in range(len(boxes)):

            confidence = float(boxes.conf[i])

            if confidence < self.conf:
                continue

            # Only person class
            class_id = int(boxes.cls[i])

            if class_id != 0:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # -----------------------------------------
            # Ignore very small detections
            # -----------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 80 or box_height < 120:
                continue

            points = keypoints.data[i].cpu().numpy()

            reading = self.is_reading(points)

            detections.append({
                "box": (x1, y1, x2, y2),
                "reading": reading,
                "confidence": confidence
            })

        return detections
