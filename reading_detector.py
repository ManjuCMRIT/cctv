from ultralytics import YOLO
import math


class ReadingDetector:

    def __init__(self, model_path="yolov8n-pose.pt", conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def is_reading(self, keypoints, box):

        # COCO keypoint indexes
        # 0 nose
        # 5 left shoulder
        # 6 right shoulder
        # 7 left elbow
        # 8 right elbow
        # 9 left wrist
        # 10 right wrist
        # 11 left hip
        # 12 right hip

        required = [0, 5, 6, 9, 10, 11, 12]

        for idx in required:
            if keypoints[idx][2] < 0.3:
                return False

        nose = keypoints[0][:2]

        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]

        left_wrist = keypoints[9][:2]
        right_wrist = keypoints[10][:2]

        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        shoulder_y = (
            left_shoulder[1] + right_shoulder[1]
        ) / 2

        hip_y = (
            left_hip[1] + right_hip[1]
        ) / 2

        # Height of person's upper body
        body_height = abs(hip_y - shoulder_y)

        if body_height < 20:
            return False

        # Head should be tilted/downward.
        # Nose being sufficiently below the shoulder line
        # is a simple first approximation.
        head_down = nose[1] > shoulder_y + body_height * 0.10

        # Hands should be relatively close to the body.
        wrist_left_distance = self.distance(left_wrist, left_hip)
        wrist_right_distance = self.distance(right_wrist, right_hip)

        hands_near_body = (
            wrist_left_distance < body_height * 1.2
            and wrist_right_distance < body_height * 1.2
        )

        return head_down and hands_near_body

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

            box_conf = float(boxes.conf[i])

            if box_conf < self.conf:
                continue

            box = tuple(boxes.xyxy[i].tolist())

            points = keypoints.data[i].cpu().numpy()

            reading = self.is_reading(points, box)

            detections.append({
                "box": box,
                "reading": reading,
                "confidence": box_conf
            })

        return detections
