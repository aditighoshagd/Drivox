"""
DriverPulse AI - Next-Gen Driver Safety & Fatigue Intelligence Dashboard
Interactive Streamlit application providing real-time vision analytics,
dynamic fatigue risk gauges, model drift tracking, and SQL session telemetry.
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime

from src.core.detector import FatigueDetector
from src.core.alert_manager import AlertManager, AlertLevel
from src.pipeline.data_pipeline import ImagePreprocessingPipeline
from src.monitoring.drift_monitor import DriftMonitor
from src.database.telemetry_db import TelemetryDatabase

# Page Configuration
st.set_page_config(
    page_title="DriverPulse AI | Enterprise Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .status-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        display: inline-block;
        margin-bottom: 12px;
    }
    .status-alert { background-color: #065f46; color: #34d399; }
    .status-warning { background-color: #78350f; color: #fbbf24; }
    .status-critical { background-color: #881337; color: #f43f5e; animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"TRIP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if "total_frames" not in st.session_state:
        st.session_state.total_frames = 0
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = []
    if "mar_history" not in st.session_state:
        st.session_state.mar_history = []
    if "fatigue_scores" not in st.session_state:
        st.session_state.fatigue_scores = []


init_session_state()

# Header
st.title("🛡️ DriverPulse AI — Fatigue & Safety Intelligence")
st.caption("Production-Grade Real-Time Computer Vision & Edge Telemetry System | Built for Automotive AI")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    driver_id = st.text_input("Driver ID", value="DRIVER-NITJ-2027")
    vehicle_id = st.text_input("Vehicle ID", value="TRUCK-VOLVO-44")

    st.subheader("Threshold Calibration")
    ear_threshold = st.slider("EAR Drowsiness Threshold", 0.15, 0.35, 0.25, 0.01)
    consec_frames = st.slider("Consecutive Closed Frames", 10, 40, 20, 1)
    mar_threshold = st.slider("MAR Yawn Threshold", 0.40, 0.90, 0.65, 0.05)
    gamma_val = st.slider("Night Vision Gamma", 0.8, 2.0, 1.3, 0.1)

    enable_audio = st.checkbox("Enable Audio Alerts", value=True)

    st.markdown("---")
    st.info("💡 **Enterprise AI Architecture:** MediaPipe 468-point 3D landmark mesh + Temporal Hysteresis + PnP Head Pose + Covariate Shift Drift Detector.")

# Singletons (Cached in Streamlit)
@st.cache_resource
def load_system_modules():
    detector = FatigueDetector(
        ear_threshold=ear_threshold,
        ear_consec_frames=consec_frames,
        mar_threshold=mar_threshold
    )
    alert_mgr = AlertManager()
    pipeline = ImagePreprocessingPipeline(gamma=gamma_val)
    drift_monitor = DriftMonitor()
    db = TelemetryDatabase()
    return detector, alert_mgr, pipeline, drift_monitor, db


detector, alert_mgr, pipeline, drift_monitor, db = load_system_modules()

# Main Tabs
tab_live, tab_telemetry, tab_fleet = st.tabs(["🎥 Live Edge Monitor", "📊 Drift & SLAs", "🗄️ SQL Fleet Database"])

with tab_live:
    col_video, col_metrics = st.columns([2, 1])

    with col_metrics:
        st.subheader("Driver Status")
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="status-badge status-alert">🟢 SYSTEM READY / ALERT</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            ear_card = st.empty()
            ear_card.markdown('<div class="metric-card"><div class="metric-title">Eye Aspect Ratio</div><div class="metric-value">0.32</div></div>', unsafe_allow_html=True)
        with m2:
            mar_card = st.empty()
            mar_card.markdown('<div class="metric-card"><div class="metric-title">Mouth Aspect Ratio</div><div class="metric-value">0.18</div></div>', unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        with m3:
            perclos_card = st.empty()
            perclos_card.markdown('<div class="metric-card"><div class="metric-title">PERCLOS (Fatigue)</div><div class="metric-value">0.0%</div></div>', unsafe_allow_html=True)
        with m4:
            fps_card = st.empty()
            fps_card.markdown('<div class="metric-card"><div class="metric-title">Edge FPS</div><div class="metric-value">34.2</div></div>', unsafe_allow_html=True)

        st.subheader("Fatigue Index")
        fatigue_prog = st.progress(0.0)

    with col_video:
        st.subheader("Real-Time Camera Stream")
        run_camera = st.checkbox("Start Live Driver Camera", value=False)
        video_placeholder = st.empty()

        if run_camera:
            if not st.session_state.session_active:
                st.session_state.session_active = True
                db.create_session(st.session_state.session_id, driver_id, vehicle_id)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Verify device permissions or test with synthetic frames.")
            else:
                frame_count = 0
                prev_time = time.time()

                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab video frame.")
                        break

                    frame_count += 1
                    curr_time = time.time()
                    fps = 1.0 / max(0.001, (curr_time - prev_time))
                    prev_time = curr_time

                    # Preprocess and Detect
                    processed = pipeline.preprocess_stream_frame(frame)
                    result, landmarks = detector.process_frame(processed)
                    alert_level = alert_mgr.evaluate(result.status, result.fatigue_score)

                    # Update Drift Monitor
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    drift_monitor.record_inference(
                        frame_luminance=float(np.mean(gray)),
                        ear_value=result.ear,
                        latency_ms=result.latency_ms,
                        is_alert=result.is_drowsy or result.is_yawning
                    )

                    # Annotate frame
                    annotated = processed.copy()
                    if landmarks:
                        # Draw eye contour
                        for idx in detector.LEFT_EYE_INDICES + detector.RIGHT_EYE_INDICES:
                            cv2.circle(annotated, landmarks[idx], 2, (0, 255, 0), -1)
                        for idx in detector.MOUTH_INDICES:
                            cv2.circle(annotated, landmarks[idx], 2, (255, 200, 0), -1)

                    # Overlay status banner
                    color = (0, 255, 0)
                    if result.status == "DROWSY":
                        color = (0, 0, 255)
                    elif result.status in ("DISTRACTED", "YAWN_WARNING"):
                        color = (0, 165, 255)

                    cv2.putText(annotated, f"STATE: {result.status}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(annotated, f"EAR: {result.ear:.2f} | MAR: {result.mar:.2f}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Convert to RGB for Streamlit
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    # Update metric cards
                    ear_card.markdown(f'<div class="metric-card"><div class="metric-title">Eye Aspect Ratio</div><div class="metric-value">{result.ear:.2f}</div></div>', unsafe_allow_html=True)
                    mar_card.markdown(f'<div class="metric-card"><div class="metric-title">Mouth Aspect Ratio</div><div class="metric-value">{result.mar:.2f}</div></div>', unsafe_allow_html=True)
                    perclos_card.markdown(f'<div class="metric-card"><div class="metric-title">PERCLOS (Fatigue)</div><div class="metric-value">{result.perclos*100:.1f}%</div></div>', unsafe_allow_html=True)
                    fps_card.markdown(f'<div class="metric-card"><div class="metric-title">Edge FPS</div><div class="metric-value">{fps:.1f}</div></div>', unsafe_allow_html=True)
                    fatigue_prog.progress(result.fatigue_score)

                    # Status badge update
                    if result.status == "DROWSY":
                        status_placeholder.markdown('<div class="status-badge status-critical">🚨 CRITICAL DROWSINESS DETECTED!</div>', unsafe_allow_html=True)
                    elif result.status == "YAWN_WARNING":
                        status_placeholder.markdown('<div class="status-badge status-warning">⚠️ YAWN / FATIGUE ADVISORY</div>', unsafe_allow_html=True)
                    elif result.status == "DISTRACTED":
                        status_placeholder.markdown('<div class="status-badge status-warning">👀 DISTRACTION DETECTED</div>', unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown('<div class="status-badge status-alert">🟢 DRIVER ALERT & ATTENTIVE</div>', unsafe_allow_html=True)

                cap.release()

with tab_telemetry:
    st.subheader("Model Drift & Inference SLA Telemetry")
    st.markdown("Real-time telemetry indicators monitoring covariate shift (environmental lighting) and physiological baseline drift.")
    metrics = drift_monitor.compute_drift_indicators()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Monitoring Status", metrics.get("status", "N/A"))
    d2.metric("P95 Latency", f"{metrics.get('latency_p95_ms', 0.0)} ms")
    d3.metric("P99 Latency", f"{metrics.get('latency_p99_ms', 0.0)} ms")
    d4.metric("Mean FPS", f"{metrics.get('mean_fps', 0.0)}")

    st.json(metrics)

with tab_fleet:
    st.subheader("Relational Fleet Telemetry & Driver Risk Scoring")
    st.markdown("Relational database analytics stored via SQLite / SQL.")
    fleet_data = db.get_fleet_risk_analytics()
    if fleet_data:
        df = pd.DataFrame(fleet_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No completed fleet trips logged in database yet. Run a driving session to record SQL telemetry.")
