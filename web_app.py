"""Render-compatible command dashboard for Gujarat Prahari AI."""

from __future__ import annotations

import os
import base64
import binascii
import hmac
import csv
from io import StringIO
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
    "severity": "low",
    "inference_ms": 0.0,
    "reconnects": 0,
    "dropped_frames": 0,
}
latest_jpeg: bytes | None = None
evidence_images: dict[str, bytes] = {}
evidence_clips: dict[str, bytes] = {}
runtime_settings = {
    "confidence": 0.30,
    "loiter_seconds": 8.0,
    "crowd_min_people": 5,
    "crowd_jump": 3,
}
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
    .side{padding:16px}.metric{padding:13px;margin-bottom:10px;border:1px solid var(--line);border-radius:12px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px}.value{font-size:22px;font-weight:800;margin-top:4px}.ok{color:var(--green)}.warn{color:var(--amber)}.danger{color:var(--red)}button,.btn{padding:10px;border:1px solid var(--cyan);border-radius:9px;background:#0f3347;color:#ecf8ff;font-weight:750;cursor:pointer;text-decoration:none}.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.count{text-align:center;padding:15px 8px;background:#0d2235;border:1px solid var(--line);border-radius:12px}.count b{display:block;font-size:27px;color:var(--cyan)}.count small{color:var(--muted)}.events{margin-top:18px;padding:18px}.event{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid var(--line)}.event:last-child{border:0}.architecture{margin-top:18px;padding:20px}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.node{background:#102a40;border:1px solid #26516f;padding:12px;border-radius:10px}.arrow{color:var(--cyan);font-size:20px}.note{color:var(--muted);line-height:1.5;margin-top:12px}.health{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.toolbar input,.toolbar select{background:#071523;color:#eef;border:1px solid var(--line);padding:9px;border-radius:8px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.evidence img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:8px}.settings{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.settings label{color:var(--muted);font-size:12px}.settings input{width:100%;margin-top:5px;background:#071523;color:#eef;border:1px solid var(--line);padding:9px;border-radius:8px}@media(max-width:800px){.grid{grid-template-columns:1fr}.counts{grid-template-columns:repeat(3,1fr)}.health,.settings{grid-template-columns:repeat(2,1fr)}}
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
   <div class="health"><div class="count"><b id="latency">0</b><small>MS</small></div><div class="count"><b id="drops">0</b><small>DROPS</small></div><div class="count"><b id="reconnects">0</b><small>RECONNECTS</small></div><div class="count"><b id="age">--</b><small>LAST FRAME</small></div></div>
  </aside>
 </div>
 <section class="card events"><div class="title">INCIDENTS & FILTERS</div><div class="toolbar"><input id="filter-camera" placeholder="Camera ID"><select id="filter-severity"><option value="">All severities</option><option>low</option><option>medium</option><option>high</option><option>critical</option></select><input id="filter-type" placeholder="Event type"><input id="filter-date" type="date"><button onclick="loadIncidents()">FILTER</button><a class="btn" id="json-export" href="/api/events">JSON</a><a class="btn" id="csv-export" href="/api/incidents.csv">CSV</a></div><div id="events"><div class="event"><span>No alert events recorded</span><span class="ok">MONITORING</span></div></div></section>
 <section class="card events"><div class="title">ALERT EVIDENCE GALLERY</div><div id="gallery" class="gallery"><span class="note">No evidence captured</span></div></section>
 <section class="card events"><div class="title">EDGE ANALYTICS SETTINGS</div><div class="settings"><label>Confidence<input id="set-confidence" type="number" min="0.1" max="0.9" step="0.05"></label><label>Loiter seconds<input id="set-loiter" type="number" min="1" max="600"></label><label>Crowd minimum<input id="set-crowd" type="number" min="2" max="100"></label><label>Crowd jump<input id="set-jump" type="number" min="1" max="50"></label></div><div class="toolbar"><button onclick="saveSettings()">APPLY TO EDGE</button></div></section>
 <section class="card architecture"><div class="title">REAL-TIME SAFETY PIPELINE</div><div class="flow"><span class="node">RTSP CCTV</span><span class="arrow">→</span><span class="node">YOLOv8 + ByteTrack</span><span class="arrow">→</span><span class="node">Temporal Rules</span><span class="arrow">→</span><span class="node">Human-Verified Alert</span></div><p class="note">This prototype provides decision support. Alerts must be verified by an authorised operator before action.</p></section>
</main>
<script>
const byId=id=>document.getElementById(id);const ui={clock:byId('clock'),camera:byId('camera'),status:byId('status'),fps:byId('fps'),detail:byId('detail'),people:byId('people'),vehicles:byId('vehicles'),objects:byId('objects'),alert:byId('alert'),events:byId('events'),feed:byId('feed-image'),latency:byId('latency'),drops:byId('drops'),reconnects:byId('reconnects'),age:byId('age'),gallery:byId('gallery')};
function tick(){ui.clock.textContent=new Date().toLocaleTimeString('en-IN',{hour12:false})}setInterval(tick,1000);tick();
async function acknowledge(){await fetch('/api/acknowledge',{method:'POST'});refresh()}
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),s=await r.json();ui.camera.textContent=s.camera_id||'CAM01';ui.status.textContent=s.status.toUpperCase();ui.status.className='value '+(s.status==='running'||s.status==='ready'?'ok':s.status==='offline'?'danger':'warn');ui.fps.textContent=Number(s.fps).toFixed(1);ui.detail.textContent=s.detail;ui.people.textContent=s.people||0;ui.vehicles.textContent=s.vehicles||0;ui.objects.textContent=s.objects||0;ui.latency.textContent=Math.round(s.inference_ms||0);ui.drops.textContent=s.dropped_frames||0;ui.reconnects.textContent=s.reconnects||0;ui.age.textContent=s.telemetry_age_seconds==null?'--':Math.round(s.telemetry_age_seconds)+'s';const active=s.alerts||[];ui.alert.textContent=active.length?(s.severity||'high').toUpperCase()+' · '+active.join(' · '):(s.acknowledged?'ALERT ACKNOWLEDGED':'NO ACTIVE ALERT');ui.alert.className='value '+(active.length?'danger':'ok');if(s.last_frame_at)ui.feed.src='/latest.jpg?t='+Date.now()}catch(e){ui.detail.textContent='Status unavailable'}}
function filterQuery(){const p=new URLSearchParams();[['camera','filter-camera'],['severity','filter-severity'],['type','filter-type'],['date','filter-date']].forEach(([key,id])=>{const value=byId(id).value.trim();if(value)p.set(key,value)});return p.toString()}
async function loadIncidents(){const q=filterQuery(),rows=await(await fetch('/api/events?'+q)).json();ui.events.innerHTML=rows.length?rows.map(e=>`<div class="event"><span>${esc(e.camera_id)} · ${esc(e.event_type)}</span><span class="${e.severity==='critical'?'danger':'warn'}">${esc(e.severity)} · ${esc(e.occurred_at)}</span></div>`).join(''):'<div class="event"><span>No matching incidents</span></div>';byId('json-export').href='/api/events?'+q;byId('csv-export').href='/api/incidents.csv?'+q}
async function loadEvidence(){const rows=await(await fetch('/api/evidence')).json();ui.gallery.innerHTML=rows.length?rows.map(e=>`<article class="evidence"><img src="${e.snapshot_url}" alt="Alert evidence"><div>${esc(e.camera_id)} · ${esc(e.severity)}</div><small>${esc(e.type)} · ${esc(e.date)} ${esc(e.time)}</small>${e.clip_url?`<div class="toolbar"><a class="btn" href="${e.clip_url}" target="_blank">PLAY EVIDENCE CLIP</a></div>`:'<div class="note">Edge clip processing…</div>'}</article>`).join(''):'<span class="note">No evidence captured</span>'}
async function loadSettings(){const s=await(await fetch('/api/settings')).json();byId('set-confidence').value=s.confidence;byId('set-loiter').value=s.loiter_seconds;byId('set-crowd').value=s.crowd_min_people;byId('set-jump').value=s.crowd_jump}
async function saveSettings(){await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence:byId('set-confidence').value,loiter_seconds:byId('set-loiter').value,crowd_min_people:byId('set-crowd').value,crowd_jump:byId('set-jump').value})});await loadSettings()}
setInterval(refresh,1500);setInterval(loadEvidence,5000);refresh();loadIncidents();loadEvidence();loadSettings();
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
        payload = dict(state)
        last_frame = payload.get("last_frame_at")
        if last_frame:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(
                    str(last_frame).replace("Z", "+00:00")
                )).total_seconds()
                payload["telemetry_age_seconds"] = round(max(0.0, age), 1)
                if age > 12:
                    payload.update(status="offline", detail="CAMERA OFFLINE - stale edge telemetry")
            except ValueError:
                payload["telemetry_age_seconds"] = None
        return jsonify(payload)


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
    decoded = None
    if encoded_frame:
        try:
            decoded = base64.b64decode(encoded_frame, validate=True)
            if len(decoded) <= 1_500_000:
                latest_jpeg = decoded
            else:
                decoded = None
        except (ValueError, binascii.Error):
            return jsonify(error="invalid frame_jpeg"), 400
    with state_lock:
        previous = set(state["edge_alerts"])
        events = list(state["events"])
        camera_id = str(payload.get("camera_id", "CAM01"))[:40]
        new_alerts = set(alerts) - previous
        severity = str(payload.get("severity", "high"))[:20].lower()
        for alert_name in new_alerts:
            evidence_id = f"{camera_id}_{int(time.time() * 1000)}"
            if decoded:
                evidence_images[evidence_id] = decoded
                while len(evidence_images) > 20:
                    evidence_images.pop(next(iter(evidence_images)))
            event = {"message": alert_name, "type": alert_name, "camera_id": camera_id,
                     "severity": severity, "time": datetime.now().strftime("%H:%M:%S"),
                     "date": datetime.now().strftime("%Y-%m-%d"),
                     "snapshot_url": f"/evidence/{evidence_id}.jpg" if evidence_id in evidence_images else None}
            events.insert(0, event)
            store.record_event(camera_id, alert_name, severity, details=event.get("snapshot_url") or "Edge AI alert")
        has_new_alert = bool(set(alerts) - previous)
        acknowledged = bool(state["acknowledged"] and alerts and not has_new_alert)
        state.update(
            status="running", detail="Secure edge analytics connected",
            camera_id=camera_id, fps=float(payload.get("fps", 0)),
            people=max(0, int(payload.get("people", 0))),
            vehicles=max(0, int(payload.get("vehicles", 0))),
            objects=max(0, int(payload.get("objects", 0))),
            severity=severity, inference_ms=float(payload.get("inference_ms", 0)),
            reconnects=max(0, int(payload.get("reconnects", 0))),
            dropped_frames=max(0, int(payload.get("dropped_frames", 0))),
            edge_alerts=alerts, alerts=[] if acknowledged else alerts,
            acknowledged=acknowledged,
            events=events[:8], last_frame_at=payload.get("sent_at"),
        )
    return jsonify(ok=True, settings=dict(runtime_settings))


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
    return jsonify(store.recent_events(
        request.args.get("limit", 50, type=int), request.args.get("entity"),
        request.args.get("camera"), request.args.get("type"),
        request.args.get("severity"), request.args.get("date"),
    ))


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        validators = {
            "confidence": (float, 0.1, 0.9), "loiter_seconds": (float, 1, 600),
            "crowd_min_people": (int, 2, 100), "crowd_jump": (int, 1, 50),
        }
        try:
            for key, (converter, minimum, maximum) in validators.items():
                if key in payload:
                    runtime_settings[key] = max(minimum, min(converter(payload[key]), maximum))
        except (TypeError, ValueError):
            return jsonify(error="invalid setting value"), 400
    return jsonify(dict(runtime_settings))


@app.get("/api/evidence")
def api_evidence():
    with state_lock:
        items = [event for event in state["events"] if event.get("snapshot_url")]
    return jsonify(items)


@app.post("/api/evidence-upload")
def upload_evidence_clip():
    if not _telemetry_authorized():
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(silent=True) or {}
    try:
        clip = base64.b64decode(payload.get("clip_mp4", ""), validate=True)
    except (ValueError, binascii.Error):
        return jsonify(error="invalid clip_mp4"), 400
    if not clip or len(clip) > 6_000_000:
        return jsonify(error="clip must be between 1 byte and 6 MB"), 413
    camera_id = str(payload.get("camera_id", "CAM01"))[:40]
    alert_type = str(payload.get("alert_type", "ALERT"))[:120]
    clip_id = f"{camera_id}_{int(time.time() * 1000)}"
    evidence_clips[clip_id] = clip
    while len(evidence_clips) > 10:
        evidence_clips.pop(next(iter(evidence_clips)))
    clip_url = f"/evidence/{clip_id}.mp4"
    with state_lock:
        for event in state["events"]:
            if event.get("camera_id") == camera_id and event.get("type") == alert_type and not event.get("clip_url"):
                event["clip_url"] = clip_url
                break
    return jsonify(ok=True, clip_url=clip_url)


@app.get("/evidence/<evidence_id>.jpg")
def evidence_image(evidence_id: str):
    image = evidence_images.get(evidence_id)
    if image is None:
        return jsonify(error="evidence not found"), 404
    return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@app.get("/evidence/<evidence_id>.mp4")
def evidence_clip(evidence_id: str):
    clip = evidence_clips.get(evidence_id)
    if clip is None:
        return jsonify(error="evidence not found"), 404
    return Response(clip, mimetype="video/mp4", headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/incidents.csv")
def export_incidents_csv():
    rows = store.recent_events(
        500, request.args.get("entity"), request.args.get("camera"),
        request.args.get("type"), request.args.get("severity"), request.args.get("date"),
    )
    output = StringIO()
    fields = ["id", "camera_id", "event_type", "entity", "severity", "occurred_at", "details"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(output.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=prahari_incidents.csv"
    })


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
