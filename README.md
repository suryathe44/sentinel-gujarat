# Gujarat Prahari AI

### Edge-first intelligent CCTV analytics for human-verified public safety

> **Har Frame Par Nazar, Har Alert Par Tez Action.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8n-00FFFF)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Render](https://img.shields.io/badge/Dashboard-Live-35E58A)](https://sentinel-gujarat.onrender.com/)

**Gujarat Prahari AI** is a working single-camera prototype developed for the
**Gujarat Police Innovation Challenge 2026 (Sentinel Gujarat)**. It consumes an
authorised live CCTV stream, detects and tracks road users, evaluates temporal
safety rules, preserves alert evidence, and sends low-bandwidth intelligence to
a cloud command dashboard.

## [Open Live Command Dashboard](https://sentinel-gujarat.onrender.com/)

## Why this approach

Monitoring every statewide CCTV stream centrally is expensive and creates
avoidable bandwidth and latency. Prahari uses **selective edge analytics**:
full-resolution video stays near the camera, while compressed previews, counts,
health metrics, and alerts flow to the command centre.

The solution uses **Model 1** as the mandatory camera-registry/GIS-ready
foundation and implements **Model 2 — Unified Viewing and Selective Analytics**
as its primary solution model.

## Working prototype

| Capability | Status |
|---|---|
| Authorised CAM01 RTSP-over-TCP ingest | Verified |
| Person and vehicle detection | Working |
| ByteTrack tracking IDs | Working |
| Restricted-zone loitering rule | Working |
| Sudden crowd rule | Working |
| Automatic alert screenshot and clip | Working |
| Edge-to-cloud annotated preview and telemetry | Working |
| Operator alert acknowledgement | Working |
| CPU inference | Observed up to ~9 FPS in one laptop demo run* |

\*Performance depends on hardware, scene, resolution, and encoding; it is not a
statewide throughput guarantee.

## Key features

- **Real-time detection:** people, bicycles, motorcycles, cars, buses, trucks,
  backpacks, handbags, and suitcases using YOLOv8n.
- **Persistent tracking:** ByteTrack IDs support dwell-time measurement.
- **Temporal alerts:** restricted-zone loitering and sudden crowd growth.
- **False-alert controls:** consecutive-frame confirmation, alert hold, and
  repeat cooldown.
- **Camera calibration:** draw and persist a resolution-independent polygon with
  the mouse.
- **Night assistance:** optional CLAHE low-light enhancement.
- **Evidence workflow:** automatic annotated snapshot, short MP4 clip, UTC event
  metadata, and append-only JSONL audit history.
- **Command dashboard:** live annotated preview, counts, inference FPS, alert
  state, recent events, and operator acknowledgement.
- **Camera health:** online/offline state, last-frame age, reconnect count,
  dropped-frame count, and measured inference latency.
- **Track visualization:** recent movement trails for every active tracked object.
- **ROI analytics:** optional `--roi-only` mode limits counts and rules to the
  calibrated restricted polygon.
- **Incident operations:** Low/Medium/High/Critical severity, camera/date/type
  filters, evidence gallery, remote thresholds, and CSV/JSON export.
- **Resilient ingest:** newest-frame buffering, stale-frame dropping,
  RTSP-over-TCP, reconnect backoff, and timestamp-discontinuity recovery.
- **Privacy-conscious design:** camera credentials remain at the edge; alerts
  remain advisory and human verified.

## System architecture

```text
┌──────────────────────────┐
│ Authorised CCTV / CAM01  │
└────────────┬─────────────┘
             │ RTSP over TCP
             ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ Latest-frame Edge Reader │────▶│ YOLOv8n Object Detection│
└──────────────────────────┘     └────────────┬─────────────┘
                                             ▼
                                ┌──────────────────────────┐
                                │ ByteTrack Tracking IDs   │
                                └────────────┬─────────────┘
                                             ▼
                                ┌──────────────────────────┐
                                │ Temporal Safety Rules    │
                                └──────┬────────────┬──────┘
                                       │            │
                              confirmed alert      live view
                                       ▼            ▼
                         ┌──────────────────┐  ┌──────────────────┐
                         │ Evidence + Audit │  │ Operator Overlay │
                         └────────┬─────────┘  └────────┬─────────┘
                                  └──────────┬──────────┘
                                             │ Compressed JPEG + JSON
                                             ▼
                                ┌──────────────────────────┐
                                │ Flask Command Dashboard  │
                                └────────────┬─────────────┘
                                             ▼
                                Human verification / acknowledge
```

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python, JavaScript, HTML, CSS, SQL |
| Detection | Ultralytics YOLOv8n, PyTorch |
| Tracking | ByteTrack |
| Video | OpenCV, FFmpeg, RTSP over TCP |
| Analytics | NumPy, PTS-based temporal rules |
| Backend | Flask, Gunicorn |
| Persistence | SQLite, JSONL evidence audit |
| Deployment | Docker, Render dashboard, edge inference |

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/suryathe44/sentinel-gujarat.git
cd sentinel-gujarat
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` locally. Never commit real credentials:

```dotenv
VIDEO_SOURCE=rtsp://encoded-email:password@camera-host:8554/stream/cam01
DASHBOARD_URL=https://sentinel-gujarat.onrender.com
DASHBOARD_TOKEN=replace-with-a-long-random-secret
```

Run the recommended single-camera CPU demo:

```bash
python cctv_analytics.py \
  --camera-id CAM01 \
  --device cpu \
  --image-size 320 \
  --confidence 0.30 \
  --loiter-seconds 8 \
  --alert-confirm-frames 2 \
  --alert-cooldown-seconds 20 \
  --display-width 1000 \
  --display-height 600 \
  --output output/final_demo.mp4
```

The application loads `.env` automatically. Explicit command-line arguments
take priority.

### Other inputs

```bash
# Laptop webcam
python cctv_analytics.py --source 0

# Saved footage
python cctv_analytics.py --source demo.mp4 --output output/result.mp4

# Headless edge execution
python cctv_analytics.py --headless --device auto
```

## Operator controls

| Key | Action |
|---|---|
| `Z` | Start restricted-zone editor |
| Left click | Add a polygon corner |
| `Enter` | Save a zone containing at least three points |
| `C` | Cancel zone editing |
| `N` | Toggle night enhancement |
| `S` | Save a manual annotated snapshot |
| `R` | Start or stop session recording |
| `A` | Trigger a labelled operator-test alert |
| `Q` / `Esc` | Finalize videos and exit safely |

Alert artifacts are written to:

```text
output/evidence/
├── alert_history.jsonl
└── YYYYMMDD_HHMMSS_alert_type/
    ├── snapshot.jpg
    └── evidence.mp4
```

Generated video, evidence, model weights, zone calibration, and `.env` secrets
are excluded from Git.

## CPU and GPU profiles

```bash
# Laptop CPU: prioritize responsiveness
python cctv_analytics.py --device cpu --image-size 320 --frame-skip 0

# NVIDIA GPU: improve resolution after installing a matching CUDA PyTorch build
python cctv_analytics.py --device cuda:0 --image-size 640
```

Measure performance on the target hardware and camera scene. Recording and
night enhancement consume additional CPU.

## Dashboard and APIs

The free Render deployment hosts the lightweight control plane; continuous
YOLO inference runs on the edge machine.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health |
| `GET /api/status` | Live camera telemetry and alert state |
| `POST /api/telemetry` | Edge preview and metrics ingest |
| `POST /api/acknowledge` | Operator acknowledgement |
| `GET /api/cameras` | Camera registry |
| `GET /api/events` | Recent stored events |
| `GET /api/evidence` | Alert snapshot and clip gallery metadata |
| `GET /api/incidents.csv` | Filtered incident-report export |
| `GET/POST /api/settings` | Runtime analytics thresholds |
| `GET/POST /api/watchlist` | Representative watchlist adapter |
| `GET /latest.jpg` | Proxy-friendly latest annotated frame |

If telemetry is older than 12 seconds, the dashboard automatically changes the
camera state to **OFFLINE**. Small alert clips (up to 6 MB) can be relayed to the
prototype gallery; full-resolution originals remain on the edge. Render Free
storage is ephemeral, so production evidence must use encrypted object storage.

For authenticated telemetry, set the same long random value as
`DASHBOARD_TOKEN` on the edge and `TELEMETRY_TOKEN` in Render. Do not run heavy
inference on Render Free.

## Scale path

The MVP deliberately proves one selected camera end to end. A production rollout
would use one isolated ingest worker per camera, batch newest frames on shared GPU
workers, and separate ingest, inference, rules, evidence, and dashboard services.
Redis Streams or Kafka can carry events; encrypted object storage can hold clips
under an approved retention policy.

Operational metrics should include input FPS, inference latency, reconnects,
dropped frames, GPU memory, end-to-end alert delay, and false-alert rate. Camera
zones and thresholds must be calibrated independently.

## Repository structure

```text
sentinel-gujarat/
├── cctv_analytics.py       # Edge detection, tracking, rules, and controls
├── safety_features.py      # Zones, evidence recorder, dashboard publisher
├── web_app.py              # Flask command dashboard and APIs
├── intelligence_store.py   # SQLite registry/watchlist/event adapter
├── sandbox_catalog.py      # Camera catalogue discovery helper
├── requirements.txt        # Full edge AI environment
├── requirements-render.txt # Lightweight dashboard environment
├── Dockerfile
├── render.yaml
└── .env.example
```

## Responsible use and limitations

- Object detection is not proof of suspicious intent. Bags are only object
  candidates; unattended-object classification requires validated owner-object
  temporal association.
- Standard YOLO weights do not reliably identify weapons, faces, number plates,
  or criminal behaviour.
- Tracking may reset after long occlusion, hard scene cuts, or reconnection.
- Perspective, low light, and crowd density affect accuracy; camera-specific
  evaluation is mandatory.
- No live VAHAN, eGujCop, AFIS, or NAFIS integration is claimed without written
  API access and legal authorisation.
- This hackathon prototype supports trained operators; it is not an autonomous
  policing or enforcement system.

## Product identity

- **Product:** Gujarat Prahari AI
- **Interface:** Prahari Command Centre
- **Primary model:** Model 2 — Unified Viewing and Selective Analytics
- **Foundation:** Model 1 — Centralised CCTV Registry and GIS-ready metadata
- **Purpose:** Human-verified public-safety decision support

---

Built for the **Gujarat Police Innovation Challenge 2026 — Sentinel Gujarat**.
