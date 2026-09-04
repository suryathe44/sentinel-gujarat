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

# The official sandbox is designed for RTSP-over-TCP. Set this before opening
# any FFmpeg-backed capture; callers can still override it explicitly.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger("prahari")

# COCO classes understood by the standard YOLOv8 weights.
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
SUSPICIOUS_OBJECT_CLASSES = {24, 26, 28}  # backpack, handbag, suitcase
DETECTION_CLASSES = sorted({PERSON_CLASS} | VEHICLE_CLASSES | SUSPICIOUS_OBJECT_CLASSES)


@dataclass(frozen=True)
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
                LOGGER.warning("Cannot open video source; retrying in %.0f seconds", retry_delay)
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            retry_delay = 2.0
            previous_pts: Optional[float] = None

            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    LOGGER.warning("Video read failed; reconnecting")
                    break
                pts_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                pts_seconds = pts_ms / 1000.0 if pts_ms > 0 else None
                if pts_seconds is not None and previous_pts is not None and pts_seconds < previous_pts:
                    self._producer_epoch += 1
                    LOGGER.info("Scene/timestamp discontinuity detected; temporal state will reset")
                previous_pts = pts_seconds if pts_seconds is not None else previous_pts
                if self._frames.full():
                    try:
                        self._frames.get_nowait()  # discard stale frame
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

        if loitering_ids:
            self.alert_until["LOITERING IN RESTRICTED ZONE"] = now + self.settings.alert_hold_seconds
        if sudden_crowd:
            self.alert_until["SUDDEN CROWD GATHERING"] = now + self.settings.alert_hold_seconds

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
        self.running = True
        self.fps_samples: deque[float] = deque(maxlen=30)
        self._last_stream_epoch = 0
        self.telemetry: dict[str, object] = {
            "people": 0, "vehicles": 0, "objects": 0, "alerts": []
        }

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _zone_for_frame(frame: np.ndarray) -> np.ndarray:
        """Default demo zone; replace normalized points for the real camera."""
        height, width = frame.shape[:2]
        normalized = [(0.55, 0.30), (0.95, 0.30), (0.95, 0.92), (0.55, 0.92)]
        return np.array([(int(x * width), int(y * height)) for x, y in normalized], np.int32)

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
                detections.append((track_id, class_id, score, (x1, y1, x2, y2)))
                if class_id == PERSON_CLASS and track_id >= 0:
                    people.append((track_id, ((x1 + x2) // 2, y2)))

        alerts, loitering_ids = self.anomalies.evaluate(people, zone, now)
        self.telemetry = {
            "people": len(people),
            "vehicles": sum(class_id in VEHICLE_CLASSES for _, class_id, _, _ in detections),
            "objects": sum(class_id in SUSPICIOUS_OBJECT_CLASSES for _, class_id, _, _ in detections),
            "alerts": alerts,
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

        if alerts:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (0, 0, 180), -1)
            cv2.putText(frame, "ALERT", (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
            cv2.putText(frame, " | ".join(alerts), (190, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        status = f"People: {len(people)}  FPS: {fps:.1f}  Device: {self.device}"
        cv2.putText(frame, status, (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        return frame

    def _write(self, frame: np.ndarray) -> None:
        if not self.settings.output:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            Path(self.settings.output).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.settings.output, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
            )
        self.writer.write(frame)

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

                started = time.perf_counter()
                result = self._infer(frame)
                elapsed = max(time.perf_counter() - started, 1e-6)
                self.fps_samples.append(1.0 / elapsed)
                annotated = self._draw_result(frame, result, self._zone_for_frame(frame), self._analytics_time())
                self._write(annotated)

                if not self.settings.headless:
                    cv2.imshow("Gujarat Prahari AI - CCTV Analytics", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            self.reader.stop()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Real-time YOLO CCTV analytics")
    parser.add_argument("--source", default="0", help="RTSP URL, video path, or webcam index")
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
