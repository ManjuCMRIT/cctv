"""
Gadget Use Monitor — v1 (video upload)

Upload a short video (instead of live webcam) and see per-child
gadget-use detection + duration, with an annotated output video to
download. Built to deploy directly on Streamlit Community Cloud from
a GitHub repo — no local install needed to run it, no live webcam
required for this stage.
"""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from detector import GadgetUseTracker

st.set_page_config(page_title="Gadget Use Monitor", layout="wide")
st.title("📵 Gadget Use Monitor — Prototype")
st.caption(
    "Upload a short video. The app detects people and phones/laptops, "
    "and tracks how long each detected child has a gadget in view."
)

with st.sidebar:
    st.header("Settings")
    frame_skip = st.slider(
        "Frame skip (higher = faster, less smooth)", 1, 5, 1,
        help="Process every Nth frame. Raise this only if processing feels "
             "too slow -- lower values catch brief gadget-use moments better.",
    )
    conf = st.slider(
        "Detection confidence", 0.1, 0.9, 0.2, 0.05,
        help="Lower catches more (including small/partly-hidden phones) but "
             "risks more false positives. Raise if you see boxes on things "
             "that aren't actually phones/laptops.",
    )
    show_debug = st.checkbox("Show per-frame detection debug info", value=False)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    if st.button("▶ Run detection"):
        with st.spinner("Loading model..."):
            tracker = GadgetUseTracker(conf=conf)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        output_path = str(Path(tempfile.gettempdir()) / "annotated_output.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        progress = st.progress(0.0, text="Processing video...")
        dt = frame_skip / fps
        frame_idx = 0
        last_annotations = []
        last_gadget_boxes = []
        start_time = time.time()
        debug_rows = []
        processed_count = 0
        gadget_seen_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip == 0:
                last_annotations = tracker.process_frame(frame, dt)
                processed_count += 1
                last_gadget_boxes = getattr(tracker, "last_gadget_detections", [])
                if last_gadget_boxes:
                    gadget_seen_count += 1
                if show_debug:
                    t_sec = frame_idx / fps
                    if last_gadget_boxes:
                        for _box, label, score in last_gadget_boxes:
                            debug_rows.append(f"t={t_sec:.2f}s — {label} (conf {score:.2f})")
                    else:
                        debug_rows.append(f"t={t_sec:.2f}s — no gadget detected")

            # Draw the phone/laptop boxes themselves (blue) so you can see
            # exactly what's being picked up, separate from the person boxes.
            for gbox, glabel, gscore in last_gadget_boxes:
                gx1, gy1, gx2, gy2 = map(int, gbox)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 140, 0), 2)
                cv2.putText(frame, f"{glabel} {gscore:.2f}", (gx1, max(20, gy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)

            for a in last_annotations:
                x1, y1, x2, y2 = map(int, a["box"])
                color = (0, 0, 255) if a["gadget_active"] else (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Child {a['track_id']}"
                if a["gadget_active"]:
                    label += f" - {a['gadget_label']}"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(frame)
            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0),
                               text=f"Processing video... {frame_idx}/{total_frames} frames")

        cap.release()
        writer.release()
        progress.empty()

        elapsed = time.time() - start_time
        st.success(f"Done in {elapsed:.1f}s.")

        st.subheader("Annotated video")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download annotated video", f, file_name="annotated_output.mp4")

        st.subheader("Per-child gadget-use summary")
        if tracker.tracks:
            for tid, tp in tracker.tracks.items():
                st.metric(f"Child {tid}", f"{tp.gadget_seconds:.1f}s")
        else:
            st.info("No people detected in this video.")

        st.caption(
            f"Processed {processed_count} frames; a phone/laptop was detected "
            f"in {gadget_seen_count} of them ({gadget_seen_count/max(processed_count,1)*100:.0f}%). "
            "If this is much lower than what you visually see in the video, "
            "try lowering 'Detection confidence' further."
        )

        if show_debug:
            with st.expander(f"Per-frame detections ({len(debug_rows)} rows)"):
                st.text("\n".join(debug_rows) if debug_rows else "No frames processed.")
else:
    st.info("Upload a video to get started. A 10-30 second clip works well for testing.")
