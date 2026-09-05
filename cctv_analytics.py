"""Gujarat Prahari AI: low-latency CCTV analytics MVP.

Detects/tracks people, vehicles and unattended-object candidates from a webcam,
video file, or RTSP stream. It raises visual alerts for restricted-zone
loitering and sudden crowd formation.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv.

    Existing exported variables always win. This makes a local `.env` work even
    when the user runs `source .env` without Bash's `set -a` export mode.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key.replace("_", "").isalnum() and not key[0].isdigit():
            os.environ.setdefault(key, value)


load_local_env()

# The official sandbox is designed for RTSP-over-TCP. Set this before opening
# any FFmpeg-backed capture; callers can still override it explicitly.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from safety_features import DashboardPublisher, EvidenceRecorder, ZoneConfig


LOGGER = logging.getLogger("prahari")

# COCO classes understood by the standard YOLOv8 weights.
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
SUSPICIOUS_OBJECT_CLASSES = {24, 26, 28}  # backpack, handbag, suitcase
DETECTION_CLASSES = sorted({PERSON_CLASS} | VEHICLE_CLASSES | SUSPICIOUS_OBJECT_CLASSES)


@dataclass
class Settings:
    """Runtime tuning knobs. Defaults favor a laptop demo."""

    source: str
    model: str = "yolov8n.pt"
    confidence: float = 0.35
    image_size: int = 640
    frame_skip: int = 0
    loiter_seconds: float = 10.0
    crowd_min_people: int = 5
    crowd_jump: int = 3
    crowd_window_seconds: float = 8.0
    alert_hold_seconds: float = 3.0
    device: str = "auto"
    output: Optional[str] = None
    headless: bool = False
    display_width: int = 1100
    display_height: int = 700
    snapshots_dir: str = "output/snapshots"
    camera_id: str = "CAM01"
    zone_file: str = "config/cam01_zone.json"
    evidence_dir: str = "output/evidence"
    evidence_seconds: float = 10.0
    alert_confirm_frames: int = 2
    alert_cooldown_seconds: float = 20.0
    enhance_low_light: bool = False
    dashboard_url: Optional[str] = None
    dashboard_token: Optional[str] = None
    roi_only: bool = False
    trail_length: int = 20


class LatestFrameReader:
    """Read continuously in a background thread and retain only the newest frame.

    Dropping old frames is intentional: a safety monitor should show *now*, not
    slowly work through an increasingly delayed RTSP buffer.
    """

    def __init__(self, source: str) -> None:
        self.source = int(source) if source.isdigit() else source
        self._frames: queue.Queue[tuple[np.ndarray, Optional[float], int]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.latest_pts_seconds: Optional[float] = None
        self.stream_epoch = 0
        self._producer_epoch = 0
        self.reconnect_count = 0
        self.dropped_frames = 0
        self.last_frame_received: Optional[float] = None
        self.connected = False

    def start(self) -> "LatestFrameReader":
        self._thread.start()
        return self

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        retry_delay = 2.0
        while not self._stop.is_set():
            self._capture = self._open()
            if not self._capture.isOpened():
                self.connected = False
                self.reconnect_count += 1
                LOGGER.warning("Cannot open video source; retrying in %.0f seconds", retry_delay)
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            retry_delay = 2.0
            self.connected = True
            previous_pts: Optional[float] = None

            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    self.connected = False
                    self.reconnect_count += 1
                    LOGGER.warning("Video read failed; reconnecting")
                    break
                pts_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                pts_seconds = pts_ms / 1000.0 if pts_ms > 0 else None
                if pts_seconds is not None and previous_pts is not None and pts_seconds < previous_pts:
                    self._producer_epoch += 1
                    LOGGER.info("Scene/timestamp discontinuity detected; temporal state will reset")
                previous_pts = pts_seconds if pts_seconds is not None else previous_pts
                self.last_frame_received = time.monotonic()
                if self._frames.full():
                    try:
                        self._frames.get_nowait()  # discard stale frame
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                self._frames.put_nowait((frame, pts_seconds, self._producer_epoch))

            self._capture.release()
            self._stop.wait(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        try:
            frame, pts_seconds, epoch = self._frames.get(timeout=timeout)
            self.latest_pts_seconds = pts_seconds
            self.stream_epoch = epoch
            return frame
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        # Do not call release() while FFmpeg may be blocked inside read().
        # Some RTSP backends deadlock when release and read run concurrently.
        self._thread.join(timeout=2)
        if not self._thread.is_alive() and self._capture is not None:
            self._capture.release()
        elif self._thread.is_alive():
            LOGGER.warning("RTSP reader is still closing; daemon cleanup will finish on exit")


class AnomalyDetector:
    """Maintain time-based state for loitering and sudden-crowd rules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.zone_entered_at: dict[int, float] = {}
        self.crowd_history: deque[tuple[float, int]] = deque()
        self.alert_until: defaultdict[str, float] = defaultdict(float)
        self.condition_frames: defaultdict[str, int] = defaultdict(int)
        self.last_activated: defaultdict[str, float] = defaultdict(lambda: float("-inf"))

    def _update_alert(self, name: str, condition: bool, now: float) -> None:
        self.condition_frames[name] = self.condition_frames[name] + 1 if condition else 0
        confirmed = self.condition_frames[name] >= self.settings.alert_confirm_frames
        cooldown_over = now - self.last_activated[name] >= self.settings.alert_cooldown_seconds
        if confirmed and cooldown_over:
            self.alert_until[name] = now + self.settings.alert_hold_seconds
            self.last_activated[name] = now

    @staticmethod
    def inside_zone(point: tuple[int, int], zone: np.ndarray) -> bool:
        return cv2.pointPolygonTest(zone, point, False) >= 0

    def evaluate(
        self,
        people: list[tuple[int, tuple[int, int]]],
        zone: np.ndarray,
        now: float,
    ) -> tuple[list[str], set[int]]:
        active_ids = {track_id for track_id, _ in people}
        loitering_ids: set[int] = set()

        for track_id, foot_point in people:
            if self.inside_zone(foot_point, zone):
                entered = self.zone_entered_at.setdefault(track_id, now)
                if now - entered >= self.settings.loiter_seconds:
                    loitering_ids.add(track_id)
            else:
                self.zone_entered_at.pop(track_id, None)

        # Remove IDs that disappeared so state cannot grow forever.
        for stale_id in set(self.zone_entered_at) - active_ids:
            self.zone_entered_at.pop(stale_id, None)

        self.crowd_history.append((now, len(people)))
        cutoff = now - self.settings.crowd_window_seconds
        while self.crowd_history and self.crowd_history[0][0] < cutoff:
            self.crowd_history.popleft()
        previous_counts = [count for _, count in list(self.crowd_history)[:-1]]
        baseline = min(previous_counts) if previous_counts else len(people)
        sudden_crowd = (
            len(people) >= self.settings.crowd_min_people
            and len(people) - baseline >= self.settings.crowd_jump
        )

        self._update_alert("LOITERING IN RESTRICTED ZONE", bool(loitering_ids), now)
        self._update_alert("SUDDEN CROWD GATHERING", sudden_crowd, now)

        alerts = [name for name, until in self.alert_until.items() if until >= now]
        return alerts, loitering_ids


class PrahariApp:
    """YOLO inference, anomaly rules, drawing, display and optional recording."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._select_device(settings.device)
        LOGGER.info("Loading %s on %s", settings.model, self.device)
        self.model = YOLO(settings.model)
        self.anomalies = AnomalyDetector(settings)
        self.reader = LatestFrameReader(settings.source)
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording_enabled = bool(settings.output)
        self.recording_path = settings.output
        self.running = True
        self.fps_samples: deque[float] = deque(maxlen=30)
        self._last_stream_epoch = 0
        self._window_ready = False
        self.zone_config = ZoneConfig(settings.zone_file)
        self.zone_editing = False
        self.zone_draft: list[tuple[float, float]] = []
        self.display_shape: tuple[int, int] = (1, 1)
        self.night_enabled = settings.enhance_low_light
        self.manual_alert_until = 0.0
        self.previous_alerts: set[str] = set()
        self.trails: defaultdict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max(2, settings.trail_length))
        )
        self.inference_ms = 0.0
        self.evidence = EvidenceRecorder(
            settings.evidence_dir, settings.camera_id, settings.evidence_seconds
        )
        self.publisher = DashboardPublisher(
            settings.dashboard_url, settings.dashboard_token, settings.camera_id
        )
        self.telemetry: dict[str, object] = {
            "people": 0, "vehicles": 0, "objects": 0, "alerts": []
        }

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _zone_for_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.zone_editing and len(self.zone_draft) >= 3:
            height, width = frame.shape[:2]
            return np.array([(int(x * width), int(y * height)) for x, y in self.zone_draft], np.int32)
        return self.zone_config.polygon(frame)

    @staticmethod
    def _enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to luminance for clearer low-light CCTV frames."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light, channel_a, channel_b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        return cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    def _infer(self, frame: np.ndarray):
        # ByteTrack supplies stable IDs needed for measuring dwell time.
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=DETECTION_CLASSES,
            conf=self.settings.confidence,
            imgsz=self.settings.image_size,
            device=self.device,
            verbose=False,
        )[0]
        return result

    def _apply_remote_settings(self) -> None:
        values = self.publisher.remote_settings
        if not values:
            return
        self.settings.confidence = max(0.1, min(float(values.get("confidence", self.settings.confidence)), 0.9))
        self.settings.loiter_seconds = max(1.0, min(float(values.get("loiter_seconds", self.settings.loiter_seconds)), 600.0))
        self.settings.crowd_min_people = max(2, min(int(values.get("crowd_min_people", self.settings.crowd_min_people)), 100))
        self.settings.crowd_jump = max(1, min(int(values.get("crowd_jump", self.settings.crowd_jump)), 50))

    def _analytics_time(self) -> float:
        """Use stream PTS for network/file feeds; webcam-only fallback uses monotonic time."""
        if self.reader.stream_epoch != self._last_stream_epoch:
            self.anomalies = AnomalyDetector(self.settings)
            self._last_stream_epoch = self.reader.stream_epoch
        return self.reader.latest_pts_seconds if self.reader.latest_pts_seconds is not None else time.monotonic()

    def _draw_result(self, frame: np.ndarray, result, zone: np.ndarray, now: float) -> np.ndarray:
        people: list[tuple[int, tuple[int, int]]] = []
        detections: list[tuple[int, int, float, tuple[int, int, int, int]]] = []

        boxes = result.boxes
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            for box, class_id, score, track_id in zip(coordinates, classes, scores, ids):
                x1, y1, x2, y2 = box
                foot_point = ((x1 + x2) // 2, y2)
                if self.settings.roi_only and not AnomalyDetector.inside_zone(foot_point, zone):
                    continue
                detections.append((track_id, class_id, score, (x1, y1, x2, y2)))
                if class_id == PERSON_CLASS and track_id >= 0:
                    people.append((track_id, foot_point))

        alerts, loitering_ids = self.anomalies.evaluate(people, zone, now)
        if time.monotonic() < self.manual_alert_until:
            alerts.append("DEMO ALERT - OPERATOR TEST")
        severity = (
            "critical" if any("LOITERING" in item for item in alerts)
            else "high" if alerts
            else "medium" if any(class_id in SUSPICIOUS_OBJECT_CLASSES for _, class_id, _, _ in detections)
            else "low"
        )
        self.telemetry = {
            "people": len(people),
            "vehicles": sum(class_id in VEHICLE_CLASSES for _, class_id, _, _ in detections),
            "objects": sum(class_id in SUSPICIOUS_OBJECT_CLASSES for _, class_id, _, _ in detections),
            "alerts": alerts,
            "severity": severity,
            "inference_ms": round(self.inference_ms, 1),
            "connected": self.reader.connected,
            "reconnects": self.reader.reconnect_count,
            "dropped_frames": self.reader.dropped_frames,
            "roi_only": self.settings.roi_only,
        }
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
        cv2.polylines(frame, [zone], True, (0, 0, 255), 3)
        cv2.putText(frame, "RESTRICTED ZONE", tuple(zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for track_id, class_id, score, (x1, y1, x2, y2) in detections:
            is_alert = track_id in loitering_ids
            color = (0, 0, 255) if is_alert else (0, 220, 0)
            label = self.model.names[class_id]
            if class_id in SUSPICIOUS_OBJECT_CLASSES:
                label = f"object:{label}"
                color = (0, 165, 255)
            text = f"{label} {score:.2f} ID:{track_id}" if track_id >= 0 else f"{label} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if track_id >= 0:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.trails[track_id].append(center)
                points = np.array(self.trails[track_id], np.int32)
                if len(points) >= 2:
                    cv2.polylines(frame, [points], False, color, 2)
        active_track_ids = {track_id for track_id, _, _, _ in detections if track_id >= 0}
        for stale_track_id in set(self.trails) - active_track_ids:
            self.trails.pop(stale_track_id, None)

        if alerts:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (0, 0, 180), -1)
            cv2.putText(frame, "ALERT", (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
            cv2.putText(frame, " | ".join(alerts), (190, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        connection = "ONLINE" if self.reader.connected else "RECONNECTING"
        status = (f"{self.settings.camera_id} {connection} | People:{len(people)} Vehicles:"
                  f"{self.telemetry['vehicles']} | FPS:{fps:.1f} Latency:{self.inference_ms:.0f}ms "
                  f"Drops:{self.reader.dropped_frames} Reconnects:{self.reader.reconnect_count}")
        cv2.putText(frame, status, (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        if self.zone_editing:
            height, width = frame.shape[:2]
            draft_pixels = [(int(x * width), int(y * height)) for x, y in self.zone_draft]
            for point in draft_pixels:
                cv2.circle(frame, point, 7, (0, 255, 255), -1)
            if len(draft_pixels) >= 2:
                cv2.polylines(frame, [np.array(draft_pixels, np.int32)], False, (0, 255, 255), 3)
            cv2.putText(frame, "ZONE EDIT: click points | ENTER save | C cancel", (15, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        return frame

    def _write(self, frame: np.ndarray) -> None:
        if not self.recording_enabled:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            if not self.recording_path:
                self.recording_path = time.strftime("output/recording_%Y%m%d_%H%M%S.mp4")
            Path(self.recording_path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.recording_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
            )
            LOGGER.info("Recording started: %s", self.recording_path)
        self.writer.write(frame)

    def _toggle_recording(self) -> None:
        self.recording_enabled = not self.recording_enabled
        if not self.recording_enabled and self.writer is not None:
            self.writer.release()
            self.writer = None
            LOGGER.info("Recording saved: %s", self.recording_path)
        elif self.recording_enabled:
            self.recording_path = time.strftime("output/recording_%Y%m%d_%H%M%S.mp4")
            LOGGER.info("Recording armed; next processed frame starts the file")

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Fit the preview inside the laptop display without changing recordings."""
        height, width = frame.shape[:2]
        scale = min(
            self.settings.display_width / width,
            self.settings.display_height / height,
            1.0,
        )
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _mouse_event(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self.zone_editing:
            return
        width, height = self.display_shape
        self.zone_draft.append((max(0.0, min(x / width, 1.0)), max(0.0, min(y / height, 1.0))))
        LOGGER.info("Zone point added (%d total)", len(self.zone_draft))

    def _save_snapshot(self, frame: np.ndarray) -> Path:
        directory = Path(self.settings.snapshots_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / time.strftime("evidence_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(str(path), frame)
        LOGGER.info("Snapshot saved: %s", path)
        return path

    def run(self) -> None:
        self.reader.start()
        processed = 0
        try:
            while self.running:
                frame = self.reader.read()
                if frame is None:
                    continue
                processed += 1
                if processed % (self.settings.frame_skip + 1):
                    continue

                working_frame = self._enhance_frame(frame) if self.night_enabled else frame
                self._apply_remote_settings()
                started = time.perf_counter()
                result = self._infer(working_frame)
                elapsed = max(time.perf_counter() - started, 1e-6)
                self.inference_ms = elapsed * 1000.0
                self.fps_samples.append(1.0 / elapsed)
                annotated = self._draw_result(
                    working_frame, result, self._zone_for_frame(working_frame), self._analytics_time()
                )
                current_alerts = set(self.telemetry["alerts"])
                for alert_name in current_alerts - self.previous_alerts:
                    self.evidence.start_alert(alert_name, annotated, self.telemetry)
                self.previous_alerts = current_alerts
                self.evidence.add_frame(annotated)
                for alert_type, clip_path in self.evidence.pop_completed():
                    self.publisher.publish_clip(alert_type, clip_path)
                fps = sum(self.fps_samples) / len(self.fps_samples)
                self.publisher.publish(self.telemetry, fps, annotated)
                self._write(annotated)

                if not self.settings.headless:
                    window_name = "Gujarat Prahari AI - CCTV Analytics"
                    if not self._window_ready:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback(window_name, self._mouse_event)
                        self._window_ready = True
                    preview = self._display_frame(annotated)
                    self.display_shape = (preview.shape[1], preview.shape[0])
                    cv2.imshow(window_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        self._save_snapshot(annotated)
                    elif key == ord("n"):
                        self.night_enabled = not self.night_enabled
                        LOGGER.info("Night enhancement: %s", "ON" if self.night_enabled else "OFF")
                    elif key == ord("a"):
                        self.manual_alert_until = time.monotonic() + self.settings.alert_hold_seconds
                    elif key == ord("r"):
                        self._toggle_recording()
                    elif key == ord("z"):
                        self.zone_editing = True
                        self.zone_draft = []
                        LOGGER.info("Zone editor ON: click at least 3 polygon points, then press Enter")
                    elif key in (10, 13) and self.zone_editing:
                        if len(self.zone_draft) >= 3:
                            self.zone_config.save(self.zone_draft)
                            self.anomalies = AnomalyDetector(self.settings)
                            self.zone_editing = False
                        else:
                            LOGGER.warning("Add at least three points before saving the zone")
                    elif key == ord("c") and self.zone_editing:
                        self.zone_editing = False
                        self.zone_draft = []
        finally:
            self.reader.stop()
            self.evidence.close()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Real-time YOLO CCTV analytics")
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"),
                        help="RTSP URL, video path, or webcam index")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=0, help="Skip N frames between inferences")
    parser.add_argument("--loiter-seconds", type=float, default=10.0)
    parser.add_argument("--crowd-min-people", type=int, default=5)
    parser.add_argument("--crowd-jump", type=int, default=3)
    parser.add_argument("--crowd-window-seconds", type=float, default=8.0)
    parser.add_argument("--alert-hold-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--output", help="Optional annotated MP4 path")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window")
    parser.add_argument("--display-width", type=int, default=1100, help="Maximum preview width")
    parser.add_argument("--display-height", type=int, default=700, help="Maximum preview height")
    parser.add_argument("--snapshots-dir", default="output/snapshots", help="Folder used by the S key")
    parser.add_argument("--camera-id", default="CAM01", help="Camera ID included in evidence and telemetry")
    parser.add_argument("--zone-file", default="config/cam01_zone.json", help="Saved restricted-zone polygon")
    parser.add_argument("--evidence-dir", default="output/evidence", help="Automatic alert evidence folder")
    parser.add_argument("--evidence-seconds", type=float, default=10.0, help="Alert clip duration")
    parser.add_argument("--alert-confirm-frames", type=int, default=2, help="Consecutive rule hits required")
    parser.add_argument("--alert-cooldown-seconds", type=float, default=20.0, help="Repeat alert cooldown")
    parser.add_argument("--enhance-low-light", action="store_true", help="Start with night enhancement enabled")
    parser.add_argument("--dashboard-url", default=os.getenv("DASHBOARD_URL"), help="Optional command dashboard URL")
    parser.add_argument("--dashboard-token", default=os.getenv("DASHBOARD_TOKEN"), help="Optional telemetry bearer token")
    parser.add_argument("--roi-only", action="store_true", help="Count and alert only inside the restricted zone")
    parser.add_argument("--trail-length", type=int, default=20, help="Tracked path length in processed frames")
    return Settings(**vars(parser.parse_args()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = PrahariApp(parse_args())

    def request_stop(_signum, _frame) -> None:
        app.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    app.run()


if __name__ == "__main__":
    main()


# Backward-compatible import for deployments created before the product rename.
SentinelApp = PrahariApp
