import cv2
import streamlit as st

from reading_detector import ReadingDetector


st.set_page_config(
    page_title="Reading Detection",
    layout="wide"
)

st.title("📖 Reading Detection - Prototype")

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded is not None:

    input_path = "reading_input.mp4"

    with open(input_path, "wb") as f:
        f.write(uploaded.read())

    if st.button("▶ Detect Reading"):

        detector = ReadingDetector()

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "reading_output.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        progress = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_number = 0

        reading_frames = 0
        total_people_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Process every frame initially
            detections = detector.process_frame(frame)

            for detection in detections:

                x1, y1, x2, y2 = map(
                    int,
                    detection["box"]
                )

                reading = detection["reading"]

                total_people_frames += 1

                if reading:
                    reading_frames += 1

                    label = "READING"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                else:

                    label = "NOT READING"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            writer.write(frame)

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

        cap.release()
        writer.release()

        st.success("Reading detection completed.")

        st.subheader("Result")

        st.video(output_path)

        if total_people_frames > 0:

            percentage = (
                reading_frames /
                total_people_frames
            ) * 100

            st.metric(
                "Reading-like frames",
                f"{percentage:.1f}%"
            )
