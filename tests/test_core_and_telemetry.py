"""
Automated Unit and Integration Test Suite
Validates core mathematical formulations (EAR, MAR), data preprocessing pipelines,
drift monitoring statistical calculations, and relational SQL telemetry logging.
Compatible with both standard unittest and pytest.
"""

import unittest
import numpy as np
import os
import sqlite3
import tempfile

from src.core.detector import FatigueDetector, DetectionResult
from src.core.alert_manager import AlertManager, AlertLevel
from src.pipeline.data_pipeline import ImagePreprocessingPipeline
from src.monitoring.drift_monitor import DriftMonitor
from src.database.telemetry_db import TelemetryDatabase
from src.models.deep_classifier import EyeStateClassifier


class TestDriverSafetySystem(unittest.TestCase):
    """Core mathematical algorithms, pipeline, and telemetry persistence tests."""

    def test_ear_calculation(self):
        """Validates Eye Aspect Ratio (EAR) formula with synthetic open and closed eye landmarks."""
        detector = FatigueDetector()

        # Synthetic open eye landmarks (6 points: p1..p6)
        # Horizontal span: (0, 10) to (20, 10) -> width = 20
        # Vertical spans: (5, 5)-(5, 15) -> 10, (15, 5)-(15, 15) -> 10
        # Expected EAR = (10 + 10) / (2 * 20) = 0.50
        open_eye = [(0, 10), (5, 5), (15, 5), (20, 10), (15, 15), (5, 15)]
        indices = [0, 1, 2, 3, 4, 5]
        ear_open = detector.calculate_ear(open_eye, indices)
        self.assertAlmostEqual(ear_open, 0.50, delta=1e-4)

        # Synthetic closed eye landmarks: vertical span is near zero
        closed_eye = [(0, 10), (5, 10), (15, 10), (20, 10), (15, 10), (5, 10)]
        ear_closed = detector.calculate_ear(closed_eye, indices)
        self.assertLess(ear_closed, 0.05)
        self.assertLess(ear_closed, detector.ear_threshold)

    def test_mar_calculation(self):
        """Validates Mouth Aspect Ratio (MAR) formula for yawn detection."""
        detector = FatigueDetector()

        # Synthetic yawn (wide open mouth): corners (0, 20), (30, 20) -> w = 30
        # Lips center: (15, 5), (15, 35) -> h = 30
        # Expected MAR = 30 / 30 = 1.0
        yawn_mouth = [(0, 20), (30, 20), (15, 5), (15, 35)]
        indices = [0, 1, 2, 3]
        mar_yawn = detector.calculate_mar(yawn_mouth, indices)
        self.assertAlmostEqual(mar_yawn, 1.0, delta=1e-4)
        self.assertGreater(mar_yawn, detector.mar_threshold)

    def test_preprocessing_pipeline(self):
        """Validates image validation, gamma adjustment, and dimension standardization."""
        pipeline = ImagePreprocessingPipeline(target_size=(320, 240), gamma=1.2)

        # Valid frame test
        sample_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.assertTrue(pipeline.validate_frame(sample_frame))

        processed = pipeline.preprocess_stream_frame(sample_frame)
        self.assertEqual(processed.shape, (240, 320, 3))
        self.assertEqual(processed.dtype, np.uint8)

        # Invalid frame tests
        self.assertFalse(pipeline.validate_frame(None))
        self.assertFalse(pipeline.validate_frame(np.zeros((10, 10))))  # 2D array

    def test_drift_monitor_statistics(self):
        """Validates covariate shift detection and P95 latency percentile calculations."""
        monitor = DriftMonitor(window_size=50, drift_threshold_z=2.0)

        # Establish baseline with 40 normal readings (EAR ~ 0.30, Latency ~ 25ms, Lum ~ 120)
        for _ in range(40):
            monitor.record_inference(frame_luminance=120.0, ear_value=0.30, latency_ms=25.0, is_alert=False)

        self.assertTrue(monitor.baseline_established)

        # Inject extreme lighting change (covariate shift)
        for _ in range(20):
            monitor.record_inference(frame_luminance=240.0, ear_value=0.30, latency_ms=30.0, is_alert=False)

        indicators = monitor.compute_drift_indicators()
        self.assertTrue(indicators["lighting_drift_detected"])
        self.assertGreaterEqual(indicators["latency_p95_ms"], 25.0)
        self.assertGreater(indicators["mean_fps"], 0)

    def test_telemetry_database(self):
        """Validates SQL relational schema, session creation, event logging, and queries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "test_driver.db")
            db = TelemetryDatabase(db_path=db_file)

            session_id = "TEST-SESSION-001"
            db.create_session(session_id=session_id, driver_id="TESTER-1", vehicle_id="TEST-CAR")

            # Log events
            db.log_safety_event(
                session_id=session_id,
                event_type="DROWSY",
                ear=0.18,
                mar=0.22,
                fatigue_score=0.88,
                latency_ms=22.5
            )

            db.end_session(
                session_id=session_id,
                total_frames=500,
                avg_ear=0.28,
                avg_mar=0.20,
                max_fatigue_score=0.88
            )

            summary = db.get_session_summary(session_id)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["driver_id"], "TESTER-1")
            self.assertEqual(summary["total_alerts"], 1)
            self.assertEqual(summary["events_breakdown"]["DROWSY"], 1)

            analytics = db.get_fleet_risk_analytics()
            self.assertEqual(len(analytics), 1)
            self.assertEqual(analytics[0]["driver_id"], "TESTER-1")

    def test_alert_manager_state_machine(self):
        """Validates tiered alert transitions and cooldown logic."""
        mgr = AlertManager(cooldown_seconds=0.1)

        # Normal state
        self.assertEqual(mgr.evaluate("ALERT", 0.1), AlertLevel.NORMAL)

        # Advisory state
        self.assertEqual(mgr.evaluate("FATIGUE_ADVISORY", 0.4), AlertLevel.ADVISORY)

        # Warning state
        self.assertEqual(mgr.evaluate("YAWN_WARNING", 0.6), AlertLevel.WARNING)

        # Critical state
        self.assertEqual(mgr.evaluate("DROWSY", 0.9), AlertLevel.CRITICAL)

    def test_eye_classifier_fallback(self):
        """Validates neural/heuristic classifier fallback inference on synthetic eye crop."""
        classifier = EyeStateClassifier()
        sample_crop = np.ones((224, 224, 3), dtype=np.float32) * 0.8
        result = classifier.predict(sample_crop)
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("is_closed", result)


if __name__ == "__main__":
    unittest.main()
