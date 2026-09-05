"""Operational helpers for zones, alert evidence and dashboard telemetry."""

from __future__ import annotations

import json
import base64
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


LOGGER = logging.getLogger("prahari.features")
DEFAULT_ZONE = [(0.55, 0.30), (0.95, 0.30), (0.95, 0.92), (0.55, 0.92)]


class ZoneConfig:
    """Persist a camera's polygon as resolution-independent coordinates."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.points = list(DEFAULT_ZONE)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            points = payload.get("normalized_points", [])
            if len(points) >= 3:
                self.points = [(float(x), float(y)) for x, y in points]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not load zone %s: %s", self.path, exc)

    def save(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            raise ValueError("A restricted zone needs at least three points")
        self.points = [(max(0.0, min(x, 1.0)), max(0.0, min(y, 1.0))) for x, y in points]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"normalized_points": self.points}, indent=2), encoding="utf-8"
        )
        LOGGER.info("Restricted zone saved: %s", self.path)

    def polygon(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        return np.array([(int(x * width), int(y * height)) for x, y in self.points], np.int32)


@dataclass
class PendingClip:
    alert_type: str
    started_at: float
    path: Path
    frames: list[np.ndarray]


class EvidenceRecorder:
    """Save alert snapshots, short clips and an append-only JSONL audit trail."""

    def __init__(self, root: str, camera_id: str, clip_seconds: float = 10.0,
                 evidence_fps: float = 5.0) -> None:
        self.root = Path(root)
        self.camera_id = camera_id
        self.clip_seconds = clip_seconds
        self.evidence_fps = evidence_fps
        self.pre_frames: deque[np.ndarray] = deque(maxlen=max(1, int(3 * evidence_fps)))
        self.pending: list[PendingClip] = []
        self.completed: deque[tuple[str, Path]] = deque()

    @staticmethod
    def _slug(value: str) -> str:
        return "_".join(value.lower().replace("/", " ").split())

    def _write_history(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "alert_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def start_alert(self, alert_type: str, frame: np.ndarray, telemetry: dict) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_dir = self.root / f"{stamp}_{self._slug(alert_type)}"
        event_dir.mkdir(parents=True, exist_ok=True)
        snapshot = event_dir / "snapshot.jpg"
        cv2.imwrite(str(snapshot), frame)
        clip = event_dir / "evidence.mp4"
        self.pending.append(PendingClip(alert_type, time.monotonic(), clip,
                                        [item.copy() for item in self.pre_frames]))
        self._write_history({
            "camera_id": self.camera_id,
            "alert_type": alert_type,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(snapshot),
            "clip": str(clip),
            "telemetry": telemetry,
            "operator_status": "pending_review",
        })
        LOGGER.warning("Evidence captured for %s: %s", alert_type, event_dir)
        return snapshot

    def add_frame(self, frame: np.ndarray) -> None:
        small = frame
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            small = cv2.resize(frame, (1280, int(height * scale)), interpolation=cv2.INTER_AREA)
        self.pre_frames.append(small.copy())
        now = time.monotonic()
        completed: list[PendingClip] = []
        for clip in self.pending:
            clip.frames.append(small.copy())
            if now - clip.started_at >= self.clip_seconds:
                self._save_clip(clip)
                completed.append(clip)
        for clip in completed:
            self.pending.remove(clip)

    def _save_clip(self, clip: PendingClip) -> None:
        if not clip.frames:
            return
        height, width = clip.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip.path), cv2.VideoWriter_fourcc(*"mp4v"), self.evidence_fps, (width, height)
        )
        for frame in clip.frames:
            writer.write(frame)
        writer.release()
        self.completed.append((clip.alert_type, clip.path))
        LOGGER.info("Evidence clip finalized: %s", clip.path)

    def pop_completed(self) -> list[tuple[str, Path]]:
        items = list(self.completed)
        self.completed.clear()
        return items

    def close(self) -> None:
        for clip in self.pending:
            self._save_clip(clip)
        self.pending.clear()


class DashboardPublisher:
    """Non-blocking, rate-limited telemetry POST to the command dashboard."""

    def __init__(self, url: Optional[str], token: Optional[str], camera_id: str) -> None:
        self.base_url = url.rstrip("/") if url else None
        self.url = self.base_url + "/api/telemetry" if self.base_url else None
        self.token = token
        self.camera_id = camera_id
        self.last_sent = 0.0
        self.busy = False
        self.remote_settings: dict = {}
        self.clip_busy = False

    def publish(self, telemetry: dict, fps: float, frame: Optional[np.ndarray] = None) -> None:
        if not self.url or self.busy or time.monotonic() - self.last_sent < 1.5:
            return
        self.last_sent = time.monotonic()
        payload = dict(telemetry, camera_id=self.camera_id, fps=round(fps, 2),
                       sent_at=datetime.now(timezone.utc).isoformat())
        if frame is not None:
            preview = frame
            height, width = preview.shape[:2]
            if width > 800:
                scale = 800 / width
                preview = cv2.resize(preview, (800, int(height * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 58])
            if ok:
                payload["frame_jpeg"] = base64.b64encode(encoded).decode("ascii")
        self.busy = True
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()

    def publish_clip(self, alert_type: str, path: Path) -> None:
        if not self.base_url or self.clip_busy or not path.exists() or path.stat().st_size > 6_000_000:
            return
        self.clip_busy = True
        threading.Thread(target=self._send_clip, args=(alert_type, path), daemon=True).start()

    def _send_clip(self, alert_type: str, path: Path) -> None:
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            payload = {"camera_id": self.camera_id, "alert_type": alert_type,
                       "clip_mp4": base64.b64encode(path.read_bytes()).decode("ascii")}
            request = urllib.request.Request(
                self.base_url + "/api/evidence-upload", data=json.dumps(payload).encode(),
                headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=15):
                pass
        except (OSError, urllib.error.URLError) as exc:
            LOGGER.debug("Dashboard clip upload unavailable: %s", exc)
        finally:
            self.clip_busy = False

    def _send(self, payload: dict) -> None:
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            request = urllib.request.Request(
                self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
                if isinstance(body.get("settings"), dict):
                    self.remote_settings = body["settings"]
        except (OSError, urllib.error.URLError) as exc:
            LOGGER.debug("Dashboard telemetry unavailable: %s", exc)
        finally:
            self.busy = False
