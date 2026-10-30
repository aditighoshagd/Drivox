"""
Model Performance and Data Drift Monitoring Subsystem
Monitors data distribution drift, lighting variations, and latency percentiles (P50, P95, P99).
Fulfills enterprise AI governance requirements for tracking edge model reliability.
"""

from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
import time


class DriftMonitor:
    """
    Real-time drift detector and telemetry analyzer for computer vision inference.
    Tracks:
      1. Luminance Drift (Covariate Shift from varying environmental illumination)
      2. EAR Baseline Drift (Physiological variance or camera angle change)
      3. Latency Metrics (P50, P95, P99 percentiles for SLA compliance)
      4. Alert Frequency Volatility
    """

    def __init__(self, window_size: int = 150, drift_threshold_z: float = 2.5):
        self.window_size = window_size
        self.drift_threshold_z = drift_threshold_z

        # Sliding telemetry windows
        self.luminance_window: deque = deque(maxlen=window_size)
        self.ear_window: deque = deque(maxlen=window_size)
        self.latency_window: deque = deque(maxlen=window_size)
        self.alert_window: deque = deque(maxlen=window_size)

        # Baseline references (established during initial 100 frames)
        self.baseline_established = False
        self.baseline_luminance_mean = 0.0
        self.baseline_luminance_std = 1.0
        self.baseline_ear_mean = 0.0
        self.baseline_ear_std = 1.0

    def record_inference(
        self,
        frame_luminance: float,
        ear_value: float,
        latency_ms: float,
        is_alert: bool
    ):
        """Records telemetry sample from a single processed frame."""
        self.luminance_window.append(frame_luminance)
        self.ear_window.append(ear_value)
        self.latency_window.append(latency_ms)
        self.alert_window.append(1 if is_alert else 0)

        # Automatically establish baseline once initial window is populated
        if not self.baseline_established and len(self.ear_window) >= min(60, self.window_size // 2):
            self._calibrate_baseline()

    def _calibrate_baseline(self):
        """Calculates baseline reference statistics from initial stable driving state."""
        self.baseline_luminance_mean = float(np.mean(self.luminance_window))
        self.baseline_luminance_std = float(max(1e-5, np.std(self.luminance_window)))
        self.baseline_ear_mean = float(np.mean(self.ear_window))
        self.baseline_ear_std = float(max(1e-5, np.std(self.ear_window)))
        self.baseline_established = True

    def compute_drift_indicators(self) -> Dict[str, Any]:
        """
        Computes statistical drift scores and inference SLAs.
        Returns dictionary formatted for enterprise monitoring dashboards and Prometheus metrics.
        """
        if len(self.ear_window) < 10:
            return {
                "status": "CALIBRATING",
                "samples_count": len(self.ear_window),
                "lighting_drift_detected": False,
                "feature_drift_detected": False,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "mean_fps": 0.0
            }

        curr_lum_mean = float(np.mean(self.luminance_window))
        curr_ear_mean = float(np.mean(self.ear_window))

        # Z-score drift calculation against calibrated baseline
        lum_z = abs(curr_lum_mean - self.baseline_luminance_mean) / self.baseline_luminance_std
        ear_z = abs(curr_ear_mean - self.baseline_ear_mean) / self.baseline_ear_std

        lighting_drift = lum_z > self.drift_threshold_z
        feature_drift = ear_z > self.drift_threshold_z

        # Latency SLAs
        latencies = list(self.latency_window)
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        mean_latency = float(np.mean(latencies))
        fps = round(1000.0 / max(1.0, mean_latency), 1)

        # Alert Rate (% of frames triggering warnings in window)
        alert_rate = float(sum(self.alert_window) / len(self.alert_window))

        return {
            "status": "DRIFT_DETECTED" if (lighting_drift or feature_drift) else "HEALTHY",
            "samples_count": len(self.ear_window),
            "baseline_ear": round(self.baseline_ear_mean, 4),
            "current_ear_mean": round(curr_ear_mean, 4),
            "ear_z_score": round(ear_z, 2),
            "feature_drift_detected": feature_drift,
            "current_luminance": round(curr_lum_mean, 2),
            "luminance_z_score": round(lum_z, 2),
            "lighting_drift_detected": lighting_drift,
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "mean_fps": fps,
            "recent_alert_rate": round(alert_rate, 3)
        }
