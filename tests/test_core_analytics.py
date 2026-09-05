"""Deterministic tests for temporal analytics, zones and evidence output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from cctv_analytics import AnomalyDetector, Settings, load_local_env
from safety_features import EvidenceRecorder, ZoneConfig


class AnomalyDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            source="0",
            loiter_seconds=2.0,
            crowd_min_people=5,
            crowd_jump=3,
            crowd_window_seconds=8.0,
            alert_confirm_frames=2,
            alert_cooldown_seconds=20.0,
        )
        self.zone = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], np.int32)

    def test_person_vehicle_and_cargo_candidates_can_loiter(self) -> None:
        detector = AnomalyDetector(self.settings)
        candidates = [(1, 0, (20, 20)), (2, 2, (30, 30)), (3, 24, (40, 40))]

        alerts, _ = detector.evaluate(
            candidates, people_count=1, zone=self.zone, now=10.0
        )
        self.assertEqual(alerts, [])
        alerts, loitering = detector.evaluate(
            candidates, people_count=1, zone=self.zone, now=12.1
        )
        self.assertEqual(alerts, [])  # first confirmed-condition frame
        alerts, loitering = detector.evaluate(
            candidates, people_count=1, zone=self.zone, now=12.2
        )

        self.assertIn("LOITERING IN RESTRICTED ZONE", alerts)
        self.assertEqual(loitering, {1, 2, 3})

    def test_sudden_crowd_uses_rolling_baseline_and_confirmation(self) -> None:
        detector = AnomalyDetector(self.settings)
        detector.evaluate([], people_count=1, zone=self.zone, now=20.0)
        alerts, _ = detector.evaluate([], people_count=5, zone=self.zone, now=21.0)
        self.assertEqual(alerts, [])
        alerts, _ = detector.evaluate([], people_count=5, zone=self.zone, now=22.0)
        self.assertIn("SUDDEN CROWD GATHERING", alerts)

    def test_disappeared_track_does_not_leave_dwell_state(self) -> None:
        detector = AnomalyDetector(self.settings)
        detector.evaluate([(7, 0, (10, 10))], 1, self.zone, 1.0)
        detector.evaluate([], 0, self.zone, 1.5)
        self.assertNotIn(7, detector.zone_entered_at)

    def test_invalid_runtime_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(source="0", confidence=1.5)

    def test_env_loader_accepts_export_and_quoted_values(self) -> None:
        import os

        key = "PRAHARI_UNIT_TEST_VALUE"
        os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(f'export {key}="loaded"\n', encoding="utf-8")
            load_local_env(env_file)
        self.assertEqual(os.environ.pop(key), "loaded")


class ZoneAndEvidenceTests(unittest.TestCase):
    def test_zone_rejects_degenerate_polygon_and_persists_valid_zone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "zone.json"
            zone = ZoneConfig(str(path))
            with self.assertRaises(ValueError):
                zone.save([(0.1, 0.1), (0.1, 0.1), (0.1, 0.1)])

            zone.save([(0.1, 0.1), (0.8, 0.1), (0.8, 0.8)])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["normalized_points"]), 3)

    def test_evidence_uses_pts_duration_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvidenceRecorder(
                temp_dir, "CAM01", clip_seconds=0.4, evidence_fps=5.0
            )
            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            recorder.add_frame(frame, now=100.0)
            snapshot = recorder.start_alert(
                "TEST / ALERT", frame, {"people": 1}, now=100.0
            )
            recorder.add_frame(frame, now=100.2)
            self.assertEqual(recorder.pop_completed(), [])
            recorder.add_frame(frame, now=100.5)
            completed = recorder.pop_completed()

            self.assertTrue(snapshot.is_file())
            self.assertEqual(len(completed), 1)
            self.assertTrue(completed[0][1].is_file())
            self.assertGreater(completed[0][1].stat().st_size, 0)
            history = Path(temp_dir) / "alert_history.jsonl"
            row = json.loads(history.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["camera_id"], "CAM01")
            self.assertEqual(row["alert_type"], "TEST / ALERT")


if __name__ == "__main__":
    unittest.main()
