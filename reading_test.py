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
        f.write(uploaded.getbuffer())

    if st.button("▶ Detect Reading"):

        detector = ReadingDetector()

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Unable to open the uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        output_path = "reading_annotated.mp4"

        # MP4 video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress = st.progress(0)

        frame_number = 0
        reading_frames = 0
        total_people_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            detections = detector.process_frame(frame)

            # Count detections
            if len(detections) > 0:
                total_people_frames += len(detections)

            for detection in detections:

                x1, y1, x2, y2 = map(
                    int,
                    detection["box"]
                )

                reading = detection["reading"]

                if reading:

                    reading_frames += 1

                    label = "READING"

                    # Bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    # Label background
                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - 35)),
                        (x1 + 150, y1),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2
                    )

                else:

                    label = "NOT READING"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - 35)),
                        (x1 + 190, y1),
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

            # Write annotated frame
            writer.write(frame)

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

        cap.release()
        writer.release()

        progress.empty()

        st.success("Detection completed successfully.")

        # --------------------------------
        # Calculate reading percentage
        # --------------------------------

        if total_people_frames > 0:

            reading_percentage = (
                reading_frames /
                total_people_frames
            ) * 100

        else:

            reading_percentage = 0.0

        # --------------------------------
        # Show result
        # --------------------------------

        st.subheader("Annotated Result")

        st.video(output_path)

        st.metric(
            "Reading-like frames",
            f"{reading_percentage:.1f}%"
        )

        # --------------------------------
        # Download button
        # --------------------------------

        with open(output_path, "rb") as video_file:

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_file,
                file_name="reading_detection_annotated.mp4",
                mime="video/mp4"
            )
