"""SQLite persistence for cameras, watchlists and time-stamped AI events.

The hackathon can use representative records locally. A production deployment
would replace this adapter with authorised Government APIs and PostgreSQL.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class IntelligenceStore:
    def __init__(self, path: str = "data/prahari.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    status TEXT NOT NULL DEFAULT 'offline'
                );
                CREATE TABLE IF NOT EXISTS watchlist (
                    registration TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'high',
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    entity TEXT,
                    severity TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    details TEXT
                );
                CREATE INDEX IF NOT EXISTS events_entity_time
                    ON events(entity, occurred_at DESC);
                """
            )

    def upsert_camera(self, camera_id: str, name: str, department: str,
                      latitude: Optional[float] = None, longitude: Optional[float] = None,
                      status: str = "online") -> None:
        with self._lock, self._connect() as database:
            database.execute(
                """INSERT INTO cameras(camera_id,name,department,latitude,longitude,status)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET
                   name=excluded.name, department=excluded.department,
                   latitude=excluded.latitude, longitude=excluded.longitude,
                   status=excluded.status""",
                (camera_id, name, department, latitude, longitude, status),
            )

    def add_watchlist(self, registration: str, category: str, reason: str,
                      severity: str = "high") -> None:
        normalized = registration.upper().replace(" ", "").replace("-", "")
        with self._lock, self._connect() as database:
            database.execute(
                """INSERT INTO watchlist(registration,category,reason,severity,active)
                   VALUES(?,?,?,?,1) ON CONFLICT(registration) DO UPDATE SET
                   category=excluded.category, reason=excluded.reason,
                   severity=excluded.severity, active=1""",
                (normalized, category, reason, severity),
            )

    def match_watchlist(self, registration: str) -> Optional[dict]:
        normalized = registration.upper().replace(" ", "").replace("-", "")
        with self._connect() as database:
            row = database.execute(
                "SELECT * FROM watchlist WHERE registration=? AND active=1", (normalized,)
            ).fetchone()
        return dict(row) if row else None

    def record_event(self, camera_id: str, event_type: str, severity: str,
                     entity: Optional[str] = None, details: str = "") -> int:
        occurred_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as database:
            cursor = database.execute(
                """INSERT INTO events(camera_id,event_type,entity,severity,occurred_at,details)
                   VALUES(?,?,?,?,?,?)""",
                (camera_id, event_type, entity, severity, occurred_at, details),
            )
            return int(cursor.lastrowid)

    def cameras(self) -> list[dict]:
        with self._connect() as database:
            return [dict(row) for row in database.execute("SELECT * FROM cameras ORDER BY camera_id")]

    def watchlist(self) -> list[dict]:
        with self._connect() as database:
            return [dict(row) for row in database.execute(
                "SELECT * FROM watchlist WHERE active=1 ORDER BY registration"
            )]

    def recent_events(self, limit: int = 50, entity: Optional[str] = None) -> list[dict]:
        limit = max(1, min(limit, 500))
        with self._connect() as database:
            if entity:
                rows = database.execute(
                    "SELECT * FROM events WHERE entity=? ORDER BY occurred_at DESC LIMIT ?",
                    (entity.upper().replace(" ", "").replace("-", ""), limit),
                )
            else:
                rows = database.execute(
                    "SELECT * FROM events ORDER BY occurred_at DESC LIMIT ?", (limit,)
                )
            return [dict(row) for row in rows]

