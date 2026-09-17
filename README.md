# Gadget Use Monitor — Prototype (v1, video upload)

First feature of the AI-Based Activity Monitoring project: detect when a
child is using a phone/laptop/tablet in an uploaded video clip, and track
how long, per child.

## Why this design

- **Video upload, not live webcam** — avoids browser-camera streaming
  complexity for this stage. Works the same whether you're testing
  locally or once it's deployed on the web.
- **YOLOv8n** — pretrained on COCO, already knows `person`, `cell phone`,
  `laptop`. No training needed for v1.
- **opencv-python-headless** — the cloud-friendly build of OpenCV (no
  GUI dependencies that Streamlit Cloud's servers don't have).

No install needed on your machine once this is deployed — see below.

## Deploying with GitHub + Streamlit Community Cloud (no local install)

1. Create a new GitHub repo and push these files to it:
   `app.py`, `detector.py`, `requirements.txt`, `.gitignore`, `README.md`
2. Go to **share.streamlit.io**, sign in with GitHub, click **New app**.
3. Pick your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. First deploy takes a few minutes (installs deps,
   downloads the ~6MB YOLOv8n weights on first run).
5. You'll get a public URL — that's your live app. Any push to the repo
   auto-redeploys it.

That's the whole loop: edit code → push to GitHub → Streamlit Cloud
rebuilds automatically. Same rhythm as your attendance project.

## Using the app

1. Open the deployed URL (or run locally with `streamlit run app.py`
   if you ever do have a machine to install on).
2. Upload a short video (10-30s, one or two people, some phone/laptop
   use, some idle time).
3. Adjust **frame skip** in the sidebar if processing feels slow —
   higher = faster but less smooth annotation.
4. Click **Run detection**. You'll get an annotated output video to
   preview/download, plus per-child gadget-use time.

## How it works

- `detector.py` — `GadgetUseTracker`: runs YOLO per frame, matches
  phone/laptop boxes to the nearest person box (proximity-based), and
  keeps a lightweight IoU tracker so each child gets a stable
  `Child N` id across frames.
- `app.py` — Streamlit UI: upload, processing loop, progress bar,
  annotated video output, and the summary panel.

## Known limitations (expected for a v1 prototype)

- Tracking is naive (IoU matching only) — crossing paths can swap IDs.
- "Tablet" isn't a distinct COCO class — usually detected as `cell
  phone` or missed.
- Gadget attribution is proximity-based, not "is this person actually
  holding it" — a hand-landmark check (MediaPipe Hands) would tighten
  this later.
- Streamlit Community Cloud's free tier is CPU-only with modest RAM —
  expect it to be noticeably slower than a local GPU run; the frame
  skip slider is there to manage that.

## Suggested next increments

1. Test on real clips, tune `conf` and the proximity margin in
   `detector.py` against what you see.
2. Add Firebase: push each gadget-use session (start, end, duration,
   child id) to Firestore, same pattern as your attendance app.
3. Once upload-based detection is solid, revisit live webcam via
   `streamlit-webrtc` (streams your browser camera to the server-side
   app over WebRTC — works even though the server has no physical
   camera).
4. Move from webcam/video-upload to an IP camera / Raspberry Pi feed
   for the hardware-integration phase.
