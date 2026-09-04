"""Render-compatible command dashboard for Gujarat Prahari AI."""

from __future__ import annotations

import os
import base64
import binascii
import hmac
import threading
import time
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, ImageDraw, ImageFont
from intelligence_store import IntelligenceStore

app = Flask(__name__)
store = IntelligenceStore(os.getenv("DATABASE_PATH", "/tmp/prahari.db"))
store.upsert_camera("CAM-01", "Demo Camera", "Police", status="ready")
state_lock = threading.Lock()
local_inference_enabled = os.getenv("RUN_LOCAL_INFERENCE", "false").lower() in {"1", "true", "yes"}
state = {
    "status": "starting" if local_inference_enabled else "ready",
    "detail": "Preparing local inference" if local_inference_enabled else "Waiting for secure edge telemetry",
    "fps": 0.0,
    "last_frame_at": None,
    "source_configured": bool(os.getenv("VIDEO_SOURCE")),
    "camera_id": "CAM01",
    "acknowledged": False,
    "people": 0,
    "vehicles": 0,
    "objects": 0,
    "alerts": [],
    "edge_alerts": [],
    "events": [],
}
latest_jpeg: bytes | None = None
worker_started = False


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Gujarat Prahari AI | Command Centre</title>
  <style>
    :root{--navy:#050d18;--panel:#0b1b2c;--line:#193b55;--cyan:#36d6e7;--red:#ff425c;--muted:#91a5b7;--green:#35e58a;--amber:#ffca58}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#10314a 0,var(--navy) 40%);color:#ecf8ff;font:15px system-ui,sans-serif;min-height:100vh}
    header{padding:18px 5vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.brand{font-size:24px;font-weight:850;letter-spacing:.5px}.brand span{color:var(--cyan)}.sub{font-size:11px;color:var(--muted);letter-spacing:1.6px;margin-top:3px}
    .live{border:1px solid #31506a;border-radius:99px;padding:7px 13px}.dot{display:inline-block;width:9px;height:9px;background:#35e58a;border-radius:50%;margin-right:8px;box-shadow:0 0 12px #35e58a}
    main{width:min(1240px,94vw);margin:22px auto}.grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}.card{background:rgba(11,27,44,.94);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 16px 45px #0005}.title{padding:13px 18px;border-bottom:1px solid var(--line);font-weight:750}.feed{aspect-ratio:16/9;background:#02070c;display:grid;place-items:center}.feed img{width:100%;height:100%;object-fit:contain}
    .side{padding:16px}.metric{padding:13px;margin-bottom:10px;border:1px solid var(--line);border-radius:12px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px}.value{font-size:22px;font-weight:800;margin-top:4px}.ok{color:var(--green)}.warn{color:var(--amber)}.danger{color:var(--red)}button{width:100%;padding:11px;border:1px solid var(--cyan);border-radius:9px;background:#0f3347;color:#ecf8ff;font-weight:750;cursor:pointer}.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.count{text-align:center;padding:15px 8px;background:#0d2235;border:1px solid var(--line);border-radius:12px}.count b{display:block;font-size:27px;color:var(--cyan)}.count small{color:var(--muted)}.events{margin-top:18px;padding:18px}.event{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid var(--line)}.event:last-child{border:0}.architecture{margin-top:18px;padding:20px}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.node{background:#102a40;border:1px solid #26516f;padding:12px;border-radius:10px}.arrow{color:var(--cyan);font-size:20px}.note{color:var(--muted);line-height:1.5;margin-top:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}.counts{grid-template-columns:repeat(3,1fr)}}
  </style>
</head>
<body>
<header><div><div class="brand">GUJARAT <span>PRAHARI AI</span></div><div class="sub">INTELLIGENT VIDEO ANALYTICS & PUBLIC SAFETY</div></div><div class="live"><span class="dot"></span>PRAHARI COMMAND CENTRE · <span id="clock">--:--:--</span></div></header>
<main>
 <div class="grid">
  <section class="card"><div class="title"><span id="camera">CAM01</span> · LIVE AI ANALYTICS</div><div class="feed"><img id="feed-image" src="/latest.jpg" alt="Analytics stream"></div></section>
  <aside class="card side">
   <div class="metric"><div class="label">Service</div><div id="status" class="value warn">STARTING</div></div>
   <div class="metric"><div class="label">Inference FPS</div><div id="fps" class="value">0.0</div></div>
   <div class="metric"><div class="label">Threat assessment</div><div id="alert" class="value ok">NO ACTIVE ALERT</div></div>
   <button id="ack" onclick="acknowledge()">ACKNOWLEDGE ALERT</button>
   <div class="counts"><div class="count"><b id="people">0</b><small>PEOPLE</small></div><div class="count"><b id="vehicles">0</b><small>VEHICLES</small></div><div class="count"><b id="objects">0</b><small>OBJECTS</small></div></div>
   <div class="metric"><div class="label">Analytics modules</div><div class="note">Detection · ByteTrack · Zone dwell · Sudden crowd · Object candidates</div></div>
   <div class="metric"><div class="label">Processing</div><div class="value">EDGE-FIRST</div><div class="note" id="detail">Connecting…</div></div>
  </aside>
 </div>
 <section class="card events"><div class="title">RECENT AI EVENTS</div><div id="events"><div class="event"><span>No alert events recorded</span><span class="ok">MONITORING</span></div></div></section>
 <section class="card architecture"><div class="title">REAL-TIME SAFETY PIPELINE</div><div class="flow"><span class="node">RTSP CCTV</span><span class="arrow">→</span><span class="node">YOLOv8 + ByteTrack</span><span class="arrow">→</span><span class="node">Temporal Rules</span><span class="arrow">→</span><span class="node">Human-Verified Alert</span></div><p class="note">This prototype provides decision support. Alerts must be verified by an authorised operator before action.</p></section>
</main>
<script>
const ui={clock:document.getElementById('clock'),camera:document.getElementById('camera'),status:document.getElementById('status'),fps:document.getElementById('fps'),detail:document.getElementById('detail'),people:document.getElementById('people'),vehicles:document.getElementById('vehicles'),objects:document.getElementById('objects'),alert:document.getElementById('alert'),events:document.getElementById('events'),feed:document.getElementById('feed-image')};
function tick(){ui.clock.textContent=new Date().toLocaleTimeString('en-IN',{hour12:false})}setInterval(tick,1000);tick();
async function acknowledge(){await fetch('/api/acknowledge',{method:'POST'});refresh()}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),s=await r.json();ui.camera.textContent=s.camera_id||'CAM01';ui.status.textContent=s.status.toUpperCase();ui.status.className='value '+(s.status==='running'||s.status==='ready'?'ok':'warn');ui.fps.textContent=Number(s.fps).toFixed(1);ui.detail.textContent=s.detail;ui.people.textContent=s.people||0;ui.vehicles.textContent=s.vehicles||0;ui.objects.textContent=s.objects||0;const active=s.alerts||[];ui.alert.textContent=active.length?active.join(' · '):(s.acknowledged?'ALERT ACKNOWLEDGED':'NO ACTIVE ALERT');ui.alert.className='value '+(active.length?'danger':'ok');if((s.events||[]).length)ui.events.innerHTML=s.events.map(e=>`<div class="event"><span>${e.message}</span><span class="danger">${e.time}</span></div>`).join('');if(s.last_frame_at)ui.feed.src='/latest.jpg?t='+Date.now()}catch(e){ui.detail.textContent='Status unavailable'}}setInterval(refresh,1500);refresh();
</script>
</body></html>
"""


def placeholder_frame(message: str) -> bytes:
    from io import BytesIO

    frame = Image.new("RGB", (1280, 720), (17, 27, 40))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=42)
    small = ImageFont.load_default(size=24)
    draw.text((70, 90), "GUJARAT PRAHARI AI", fill=(54, 214, 231), font=font)
    draw.text((70, 315), message, fill=(235, 242, 247), font=small)
    draw.text((70, 370), "Configure a secure edge camera to begin analytics", fill=(155, 178, 197), font=small)
    output = BytesIO()
    frame.save(output, format="JPEG", quality=82)
    return output.getvalue()


def analytics_worker() -> None:
    global latest_jpeg
    source = os.getenv("VIDEO_SOURCE", "").strip()
    if not local_inference_enabled:
        latest_jpeg = placeholder_frame("Dashboard ready - start the secure edge analytics client")
        with state_lock:
            state.update(status="ready", detail="Waiting for secure edge telemetry")
        return
    if not source:
        latest_jpeg = placeholder_frame("RUN_LOCAL_INFERENCE needs VIDEO_SOURCE")
        with state_lock:
            state.update(status="error", detail="VIDEO_SOURCE is missing")
        return

    # Heavy AI dependencies are imported only when a camera is configured. This
    # keeps the public control-plane dashboard small enough for Render Free.
    # Full inference uses requirements.txt on an edge device or a larger service.
    import cv2
    from cctv_analytics import PrahariApp, Settings

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
        engine = PrahariApp(settings)
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
            output = engine._draw_result(frame, result, engine._zone_for_frame(frame), engine._analytics_time())
            ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                latest_jpeg = encoded.tobytes()
            with state_lock:
                telemetry = engine.telemetry
                previous_alerts = set(state["alerts"])
                current_alerts = list(telemetry["alerts"])
                events = list(state["events"])
                for alert_name in set(current_alerts) - previous_alerts:
                    events.insert(0, {"message": alert_name, "time": datetime.now().strftime("%H:%M:%S")})
                state.update(
                    fps=sum(engine.fps_samples) / len(engine.fps_samples),
                    last_frame_at=datetime.now(timezone.utc).isoformat(),
                    people=telemetry["people"], vehicles=telemetry["vehicles"],
                    objects=telemetry["objects"], alerts=current_alerts, events=events[:8],
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


def _telemetry_authorized() -> bool:
    expected = os.getenv("TELEMETRY_TOKEN", "")
    if not expected:
        return True
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    return hmac.compare_digest(supplied, expected)


@app.post("/api/telemetry")
def api_telemetry():
    """Receive low-bandwidth results from an authorised edge inference client."""
    global latest_jpeg
    if not _telemetry_authorized():
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(silent=True) or {}
    alerts = [str(item)[:120] for item in payload.get("alerts", [])][:5]
    encoded_frame = payload.get("frame_jpeg")
    if encoded_frame:
        try:
            decoded = base64.b64decode(encoded_frame, validate=True)
            if len(decoded) <= 1_500_000:
                latest_jpeg = decoded
        except (ValueError, binascii.Error):
            return jsonify(error="invalid frame_jpeg"), 400
    with state_lock:
        previous = set(state["edge_alerts"])
        events = list(state["events"])
        camera_id = str(payload.get("camera_id", "CAM01"))[:40]
        for alert_name in set(alerts) - previous:
            event = {"message": alert_name, "time": datetime.now().strftime("%H:%M:%S")}
            events.insert(0, event)
            store.record_event(camera_id, alert_name, "high", details="Edge AI alert")
        has_new_alert = bool(set(alerts) - previous)
        acknowledged = bool(state["acknowledged"] and alerts and not has_new_alert)
        state.update(
            status="running", detail="Secure edge analytics connected",
            camera_id=camera_id, fps=float(payload.get("fps", 0)),
            people=max(0, int(payload.get("people", 0))),
            vehicles=max(0, int(payload.get("vehicles", 0))),
            objects=max(0, int(payload.get("objects", 0))),
            edge_alerts=alerts, alerts=[] if acknowledged else alerts,
            acknowledged=acknowledged,
            events=events[:8], last_frame_at=payload.get("sent_at"),
        )
    return jsonify(ok=True)


@app.post("/api/acknowledge")
def api_acknowledge():
    with state_lock:
        state["acknowledged"] = True
        state["alerts"] = []
        state["events"] = [{"message": "Alert acknowledged by operator",
                            "time": datetime.now().strftime("%H:%M:%S")}] + list(state["events"])
    return jsonify(ok=True)


@app.get("/api/cameras")
def api_cameras():
    return jsonify(store.cameras())


@app.route("/api/watchlist", methods=["GET", "POST"])
def api_watchlist():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        required = ("registration", "category", "reason")
        if any(not payload.get(field) for field in required):
            return jsonify(error="registration, category and reason are required"), 400
        store.add_watchlist(payload["registration"], payload["category"], payload["reason"],
                            payload.get("severity", "high"))
        return jsonify(ok=True), 201
    return jsonify(store.watchlist())


@app.get("/api/events")
def api_events():
    return jsonify(store.recent_events(request.args.get("limit", 50, type=int), request.args.get("entity")))


@app.get("/video_feed")
def video_feed():
    ensure_worker()

    def frames():
        while True:
            image = latest_jpeg or placeholder_frame("Starting analytics worker...")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image + b"\r\n"
            time.sleep(0.12)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/latest.jpg")
def latest_frame():
    """Proxy-friendly latest-frame endpoint used by the Render dashboard."""
    ensure_worker()
    image = latest_jpeg or placeholder_frame("Starting analytics worker...")
    return Response(image, mimetype="image/jpeg", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), threaded=True)
