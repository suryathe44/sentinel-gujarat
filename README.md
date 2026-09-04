# Gujarat Prahari AI — Intelligent Video Analytics & Public Safety Platform

**Tagline:** *Har Frame Par Nazar, Har Alert Par Tez Action.*

Gujarat Prahari AI accepts a webcam, phone camera stream, video file, or RTSP CCTV feed and provides:

- YOLOv8 detection and ByteTrack tracking for people and vehicles
- backpack, handbag, and suitcase detection as suspicious-object candidates
- restricted-zone loitering alerts based on a stable track ID and dwell time
- sudden-crowd alerts based on a short rolling people-count baseline
- bounding boxes, confidence labels, a bold alert banner, FPS, and optional MP4 recording
- SQLite-backed camera registry, representative vehicle watchlist, and event-history APIs

> Important: an object class alone is not proof of suspicious intent. Treat these
> results as decision support for a human operator, not automatic enforcement.

## 1. Laptop setup

Python 3.10 or 3.11 is recommended.

```bash
cd sentinel-gujarat
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads `yolov8n.pt`. For an offline demo, run it once before
the event and keep the downloaded weights available.

## 2. Run and test

Start with a webcam:

```bash
python cctv_analytics.py --source 0 --loiter-seconds 5
```

Then test a saved video:

```bash
python cctv_analytics.py --source demo.mp4 --output output/demo_result.mp4
```

Finally use the camera's RTSP URL (quote it so shell characters are safe):

```bash
python cctv_analytics.py --source "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1"
```

### Official Sentinel sandbox

Discover cameras from the catalogue every time instead of hard-coding stream paths:

```bash
python sandbox_catalog.py --catalog "http://SANDBOX_HOST/api/ingest"
python sandbox_catalog.py --catalog "http://SANDBOX_HOST/api/ingest" --camera CAM01
```

Pass the selected catalogue RTSP URL to `--source`. The reader forces RTSP over
TCP, tolerates initial decoder warnings, drops stale frames, reconnects with
exponential backoff (2-30 seconds), and drives temporal analytics from stream PTS.
When PTS moves backwards at a loop/reboot discontinuity, dwell/crowd state resets.

The preview automatically fits inside a laptop display while the output recording
keeps the camera's full resolution. Press `s` to save an annotated evidence image
under `output/snapshots`. Press `q` or Escape to close the stream and finalize the
MP4 recording. Do not put a real RTSP password in source control, screenshots, or
the submission document.

### Single-camera operator controls

- `Z`: start drawing a restricted zone; left-click at least three corners, then
  press Enter to save it in `config/cam01_zone.json`. Press `C` to cancel.
- `N`: toggle CLAHE low-light enhancement for night footage.
- `S`: save a manual full-resolution annotated snapshot.
- `R`: start/stop an annotated session recording.
- `A`: trigger a short labelled operator-test alert for presentation rehearsal.
- `Q` or Escape: safely stop, finalize every video, and exit.

Real anomaly alerts automatically create a timestamped directory under
`output/evidence/` containing `snapshot.jpg` and `evidence.mp4`. An append-only
`alert_history.jsonl` records camera ID, UTC time, alert type, paths, counts, and
the initial `pending_review` operator status. Alert confirmation and cooldown
reduce one-frame false alarms and repeated notifications.

Recommended CAM01 demo command:

```bash
source .venv/bin/activate
source .env
python cctv_analytics.py --source "$VIDEO_SOURCE" --camera-id CAM01 \
  --device cpu --image-size 320 --confidence 0.30 --loiter-seconds 8 \
  --alert-confirm-frames 2 --alert-cooldown-seconds 20 \
  --display-width 1000 --display-height 600 --output output/final_demo.mp4
```

### Send edge results to the Render dashboard

Copy `.env.example` values into the local `.env`, keeping the existing private
`VIDEO_SOURCE`. Set `DASHBOARD_URL` to the Render service and generate a long
random `DASHBOARD_TOKEN`. In Render Environment settings, set `TELEMETRY_TOKEN`
to exactly the same value. The laptop then sends compressed annotated previews,
counts, FPS, camera ID, and alert names; raw RTSP credentials never leave the
edge machine. Leave `RUN_LOCAL_INFERENCE` unset on Render Free. The dashboard's
**ACKNOWLEDGE ALERT** button suppresses the current alert until it clears or a new
alert arrives.

## 3. Calibrate for the actual camera

Edit `_zone_for_frame()` in `cctv_analytics.py`. The four `(x, y)` pairs are
fractions of image width and height, so `(0.55, 0.30)` means 55% from the left
and 30% from the top. Place them around the restricted area visible in the feed.

Useful demo tuning:

```bash
python cctv_analytics.py --source demo.mp4 \
  --loiter-seconds 8 --crowd-min-people 5 --crowd-jump 3 \
  --crowd-window-seconds 8 --confidence 0.35
```

The crowd rule fires when the current count is at least `crowd-min-people` and
has increased by `crowd-jump` relative to the lowest recent count in the rolling
window. Use a longer loiter time (30–120 seconds) in a real deployment.

## CPU and GPU modes

- Automatic: `--device auto` chooses CUDA when available, otherwise CPU.
- CPU: use `--device cpu --image-size 416 --frame-skip 1`. YOLOv8n is the best
  starting model; reducing resolution improves speed but may miss small objects.
- NVIDIA GPU: install the PyTorch build matching the installed CUDA driver, then
  use `--device cuda:0 --image-size 640`. Ultralytics selects the supported
  execution optimizations for the installed runtime.
- Verify CUDA before the demo with:
  `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

For a fair FPS test, disable recording because video encoding consumes CPU.

## Four-day build roadmap

### Day 1 — Working detection pipeline

1. Create the environment and test the webcam.
2. Obtain a legal test RTSP feed from the camera owner and confirm reconnects.
3. Record two short demo clips: normal activity and a staged violation.
4. Draw the restricted zone and tune confidence for that camera angle.

### Day 2 — Anomalies and evidence

1. Stage one loitering and one sudden-crowd scenario.
2. Tune timers/counts and confirm track IDs remain reasonably stable.
3. Save annotated output clips and collect measured FPS/latency.
4. Add optional webhook/SMS integration only after the on-screen alerts work.

### Day 3 — Reliability and presentation

1. Run a 30–60 minute soak test and note reconnect behavior and false alerts.
2. Test low light, partial occlusion, empty scenes, and crowded scenes.
3. Prepare an architecture slide, a 90-second demo, limitations, and metrics.
4. Keep a prerecorded result as demo backup if venue networking fails.

### Day 4 — Submission polish

1. Freeze dependencies and back up code, weights, slides, and videos locally.
2. Rehearse: problem (15s), live detection (30s), alerts (30s), scale/impact (15s).
3. Clearly state that alerts require human verification and footage retention is controlled.
4. Submit early, then make only low-risk polish changes.

## Architecture

```text
RTSP cameras
    │
    ▼
Latest-frame reader (per camera; stale frames dropped, reconnect enabled)
    │
    ▼
YOLOv8n detector ──► ByteTrack IDs
    │                    │
    ├── people count     ├── zone entry time ──► loitering rule
    ├── vehicles         └── rolling counts ───► crowd rule
    └── object candidates
              │
              ▼
Alert/event layer ──► overlay + operator display + optional recording
              │
              └── future: API/webhook, database, evidence snapshot, control room
```

## Production scaling notes

- Run each camera capture independently so a dead stream cannot block others.
- On one GPU, batch the newest frame from several cameras into one inference call;
  benchmark the camera count and enforce a maximum end-to-end latency.
- Separate ingest, inference, event processing, and dashboard services for a large
  deployment. Redis Streams/Kafka can carry events; object storage can hold short,
  encrypted evidence clips according to an approved retention policy.
- Use camera IDs and timestamps in all events. Add health metrics for input FPS,
  inference latency, reconnects, queue drops, GPU memory, and alert rate.
- Replace the demo rules with camera-specific calibrated zones and thresholds.
  Evaluate precision/recall on consented, locally representative footage.
- Secure RTSP credentials in a secrets manager, isolate cameras on a network VLAN,
  encrypt traffic/storage, use role-based access, maintain audit logs, blur faces
  where identification is unnecessary, and keep a human in the loop.

## Deploy the dashboard on Render

The repository includes `Dockerfile`, `render.yaml`, and `web_app.py`. In Render,
create a Blueprint from this repository. The service exposes `/health`, `/api/status`,
and a browser dashboard at `/`. Set `VIDEO_SOURCE` as a secret environment variable;
never commit an RTSP username or password.

The free Render instance deploys the lightweight command-center dashboard without
loading PyTorch. Continuous YOLO inference needs more memory/CPU and the dependencies
in `requirements.txt`, so use an edge device or a larger inference service before
setting `VIDEO_SOURCE`. Public-cloud RTSP also requires a camera endpoint reachable
from the internet; for real police deployment, run inference at the camera edge and
send only alerts to the cloud dashboard.

## Integration APIs

The command service includes functional JSON endpoints for the integration layer:

- `GET /api/cameras` - onboarded camera inventory and health
- `GET /api/watchlist` - active representative watchlist records
- `POST /api/watchlist` - add/update a representative registration record
- `GET /api/events?entity=GJ01AB1234` - alert history and route evidence for an entity

These endpoints deliberately use representative SQLite data. Do not claim live VAHAN,
eGujCop, AFIS, or NAFIS integration without written API access and authorisation.

## Submission-ready technical summary

**Tech Stack:** Python, OpenCV, Ultralytics YOLOv8n, PyTorch, ByteTrack, NumPy,
RTSP/FFmpeg, and CUDA FP16 acceleration when an NVIDIA GPU is available.

**Workflow:** Each CCTV stream is read asynchronously with a latest-frame buffer
to prevent latency buildup. YOLOv8n detects people, vehicles, and selected object
candidates; ByteTrack assigns persistent IDs. A temporal rules engine measures
person dwell time inside a camera-specific polygon and compares live people count
with a rolling baseline to detect rapid crowd formation. The operator view renders
confidence-labelled boxes, restricted zones, status metrics, and a high-visibility
alert banner. Alerts are advisory and remain subject to human verification.

**Scalability:** The MVP uses one independent pipeline per camera. A production
version can batch the latest frames from multiple cameras on shared GPUs and split
video ingest, inference, event evaluation, and dashboard delivery into services.
Bounded latest-frame queues keep latency predictable under load. Centralized health
metrics, camera-specific calibration, encrypted evidence storage, access controls,
and retention policies support reliable and responsible control-room deployment.

## Honest limitations

- Standard YOLO weights recognize object categories, not criminal intent, weapons
  reliably, or a truly unattended bag. Those require curated local data, validation,
  and temporal association with an owner.
- Tracking can reset after long occlusion or an RTSP reconnect, resetting dwell time.
- Perspective affects crowd counts; camera-specific calibration is essential.
- This is a hackathon MVP, not a certified autonomous policing system.

## Product identity

- **Product:** Gujarat Prahari AI
- **Operations UI:** Prahari Command Centre
- **Purpose:** human-verified public-safety decision support
- **Tagline:** Har Frame Par Nazar, Har Alert Par Tez Action.
