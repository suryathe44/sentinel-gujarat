"""Render-compatible web dashboard for Sentinel Gujarat."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

from cctv_analytics import SentinelApp, Settings


app = Flask(__name__)
state_lock = threading.Lock()
state = {
    "status": "starting",
    "detail": "Preparing Sentinel Gujarat",
    "fps": 0.0,
    "last_frame_at": None,
    "source_configured": bool(os.getenv("VIDEO_SOURCE")),
}
latest_jpeg: bytes | None = None
worker_started = False


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sentinel Gujarat | AI Command Center</title>
  <style>
    :root{--navy:#06111f;--panel:#0b1b2c;--line:#193b55;--cyan:#36d6e7;--red:#ff425c;--muted:#91a5b7}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#10314a 0,var(--navy) 40%);color:#ecf8ff;font:15px system-ui,sans-serif;min-height:100vh}
    header{padding:22px 5vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.brand{font-size:24px;font-weight:850;letter-spacing:.5px}.brand span{color:var(--cyan)}
    .live{border:1px solid #31506a;border-radius:99px;padding:7px 13px}.dot{display:inline-block;width:9px;height:9px;background:#35e58a;border-radius:50%;margin-right:8px;box-shadow:0 0 12px #35e58a}
    main{width:min(1180px,92vw);margin:26px auto}.grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}.card{background:rgba(11,27,44,.92);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 16px 45px #0005}.title{padding:14px 18px;border-bottom:1px solid var(--line);font-weight:750}.feed{aspect-ratio:16/9;background:#02070c;display:grid;place-items:center}.feed img{width:100%;height:100%;object-fit:contain}
    .side{padding:18px}.metric{padding:15px;margin-bottom:12px;border:1px solid var(--line);border-radius:12px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1px}.value{font-size:23px;font-weight:800;margin-top:5px}.ok{color:#35e58a}.warn{color:#ffca58}.architecture{margin-top:20px;padding:20px}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.node{background:#102a40;border:1px solid #26516f;padding:12px;border-radius:10px}.arrow{color:var(--cyan);font-size:20px}.note{color:var(--muted);line-height:1.6;margin-top:15px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header><div class="brand">SENTINEL <span>GUJARAT</span></div><div class="live"><span class="dot"></span>AI COMMAND CENTER</div></header>
<main>
 <div class="grid">
  <section class="card"><div class="title">CAM-01 · LIVE ANALYTICS</div><div class="feed"><img src="/video_feed" alt="Analytics stream"></div></section>
  <aside class="card side">
   <div class="metric"><div class="label">Service</div><div id="status" class="value warn">STARTING</div></div>
   <div class="metric"><div class="label">Inference FPS</div><div id="fps" class="value">0.0</div></div>
   <div class="metric"><div class="label">Analytics</div><div class="value ok">ACTIVE</div><div class="note">People · Vehicles · Loitering · Sudden crowd</div></div>
   <div class="metric"><div class="label">Processing</div><div class="value">EDGE-FIRST</div><div class="note" id="detail">Connecting…</div></div>
  </aside>
 </div>
 <section class="card architecture"><div class="title">REAL-TIME SAFETY PIPELINE</div><div class="flow"><span class="node">RTSP CCTV</span><span class="arrow">→</span><span class="node">YOLOv8 + ByteTrack</span><span class="arrow">→</span><span class="node">Temporal Rules</span><span class="arrow">→</span><span class="node">Human-Verified Alert</span></div><p class="note">This prototype provides decision support. Alerts must be verified by an authorised operator before action.</p></section>
</main>
<script>async function refresh(){try{const r=await fetch('/api/status'),s=await r.json();status.textContent=s.status.toUpperCase();status.className='value '+(s.status==='running'?'ok':'warn');fps.textContent=Number(s.fps).toFixed(1);detail.textContent=s.detail}catch(e){detail.textContent='Status unavailable'}}setInterval(refresh,2000);refresh()</script>
</body></html>
"""


def placeholder_frame(message: str) -> bytes:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (17, 27, 40)
    cv2.putText(frame, "SENTINEL GUJARAT", (70, 120), cv2.FONT_HERSHEY_DUPLEX, 1.7, (231, 214, 54), 3)
    cv2.putText(frame, message, (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 242, 247), 2)
    cv2.putText(frame, "Set VIDEO_SOURCE in Render to an RTSP URL or video URL", (70, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 178, 197), 2)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if ok else b""


def analytics_worker() -> None:
    global latest_jpeg
    source = os.getenv("VIDEO_SOURCE", "").strip()
    if not source:
        latest_jpeg = placeholder_frame("Cloud dashboard is ready - camera source is not configured")
        with state_lock:
            state.update(status="ready", detail="Waiting for VIDEO_SOURCE")
        return

    settings = Settings(
        source=source,
        model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        device=os.getenv("DEVICE", "cpu"),
        image_size=int(os.getenv("IMAGE_SIZE", "416")),
        frame_skip=int(os.getenv("FRAME_SKIP", "1")),
        loiter_seconds=float(os.getenv("LOITER_SECONDS", "10")),
        crowd_min_people=int(os.getenv("CROWD_MIN_PEOPLE", "5")),
        crowd_jump=int(os.getenv("CROWD_JUMP", "3")),
        headless=True,
    )
    try:
        engine = SentinelApp(settings)
        engine.reader.start()
        with state_lock:
            state.update(status="running", detail="Live AI inference connected")
        count = 0
        while True:
            frame = engine.reader.read()
            if frame is None:
                continue
            count += 1
            if count % (settings.frame_skip + 1):
                continue
            started = time.perf_counter()
            result = engine._infer(frame)
            engine.fps_samples.append(1.0 / max(time.perf_counter() - started, 1e-6))
            output = engine._draw_result(frame, result, engine._zone_for_frame(frame), time.monotonic())
            ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                latest_jpeg = encoded.tobytes()
            with state_lock:
                state.update(
                    fps=sum(engine.fps_samples) / len(engine.fps_samples),
                    last_frame_at=datetime.now(timezone.utc).isoformat(),
                )
    except Exception as exc:
        latest_jpeg = placeholder_frame("Analytics worker needs attention")
        with state_lock:
            state.update(status="error", detail=f"{type(exc).__name__}: {exc}")


def ensure_worker() -> None:
    global worker_started
    if not worker_started:
        worker_started = True
        threading.Thread(target=analytics_worker, name="analytics", daemon=True).start()


@app.get("/")
def index():
    ensure_worker()
    return render_template_string(PAGE)


@app.get("/health")
def health():
    ensure_worker()
    return jsonify(ok=True), 200


@app.get("/api/status")
def api_status():
    ensure_worker()
    with state_lock:
        return jsonify(dict(state))


@app.get("/video_feed")
def video_feed():
    ensure_worker()

    def frames():
        while True:
            image = latest_jpeg or placeholder_frame("Starting analytics worker...")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image + b"\r\n"
            time.sleep(0.12)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), threaded=True)
