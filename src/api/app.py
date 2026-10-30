"""
Enterprise REST API for Driver Fatigue Detection & Fleet Intelligence
Built with FastAPI. Exposes endpoints for real-time edge frame inference,
session lifecycle management, SQL telemetry queries, and drift monitoring.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import time
import os

from src.core.detector import FatigueDetector, DetectionResult
from src.core.alert_manager import AlertManager, AlertLevel
from src.pipeline.data_pipeline import ImagePreprocessingPipeline
from src.monitoring.drift_monitor import DriftMonitor
from src.database.telemetry_db import TelemetryDatabase

app = FastAPI(
    title="Drivox - Driver Fatigue & Safety Intelligence API",
    description="Production-grade AI microservice for edge and cloud vehicle safety telemetry.",
    version="2.0.0"
)

# Enable CORS for cross-platform integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Singletons
detector = FatigueDetector()
alert_mgr = AlertManager()
pipeline = ImagePreprocessingPipeline()
drift_monitor = DriftMonitor()
db = TelemetryDatabase()

SERVER_START_TIME = time.time()


# Request / Response Schemas
class SessionStartRequest(BaseModel):
    session_id: str = Field(..., example="SESS-20261031-01")
    driver_id: str = Field(..., example="DRIVER-9842")
    vehicle_id: str = Field(default="FLEET-TRUCK-07", example="FLEET-TRUCK-07")


class SessionEndRequest(BaseModel):
    session_id: str
    total_frames: int
    avg_ear: float
    avg_mar: float
    max_fatigue_score: float


class FrameInferenceResponse(BaseModel):
    face_detected: bool
    status: str
    fatigue_score: float
    ear: float
    left_ear: float
    right_ear: float
    mar: float
    perclos: float
    is_drowsy: bool
    is_yawning: bool
    is_distracted: bool
    alert_level: str
    head_pose: Dict[str, float]
    latency_ms: float
    drift_status: str


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for container orchestrators (Kubernetes / Docker)."""
    return {
        "status": "HEALTHY",
        "service": "Drivox",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "api_version": "2.0.0"
    }


@app.post("/api/v1/predict/frame", response_model=FrameInferenceResponse, tags=["Inference"])
async def predict_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Performs real-time fatigue inference on an uploaded image frame.
    Preprocesses frame, detects landmarks, computes EAR/MAR/PERCLOS, and updates drift monitor.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or not pipeline.validate_frame(frame):
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    # 1. Feature Preprocessing
    processed_frame = pipeline.preprocess_stream_frame(frame)

    # 2. Fatigue Detection Engine
    result, landmarks = detector.process_frame(processed_frame)

    # 3. Alert Manager Evaluation
    alert_level = alert_mgr.evaluate(result.status, result.fatigue_score)

    # 4. Telemetry and Drift Monitoring
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray))
    drift_monitor.record_inference(
        frame_luminance=mean_luminance,
        ear_value=result.ear,
        latency_ms=result.latency_ms,
        is_alert=result.is_drowsy or result.is_yawning or result.is_distracted
    )
    drift_report = drift_monitor.compute_drift_indicators()

    # 5. Database Event Logging if an active session is provided
    if session_id and (result.is_drowsy or result.is_yawning or result.is_distracted):
        db.log_safety_event(
            session_id=session_id,
            event_type=result.status,
            ear=result.ear,
            mar=result.mar,
            fatigue_score=result.fatigue_score,
            latency_ms=result.latency_ms,
            head_pose=result.head_pose
        )

    return FrameInferenceResponse(
        face_detected=result.face_detected,
        status=result.status,
        fatigue_score=result.fatigue_score,
        ear=result.ear,
        left_ear=result.left_ear,
        right_ear=result.right_ear,
        mar=result.mar,
        perclos=result.perclos,
        is_drowsy=result.is_drowsy,
        is_yawning=result.is_yawning,
        is_distracted=result.is_distracted,
        alert_level=alert_level.name,
        head_pose=result.head_pose,
        latency_ms=result.latency_ms,
        drift_status=drift_report["status"]
    )


@app.post("/api/v1/session/start", tags=["Session Lifecycle"])
def start_session(payload: SessionStartRequest):
    """Initializes a new driver monitoring session in the SQL database."""
    db.create_session(
        session_id=payload.session_id,
        driver_id=payload.driver_id,
        vehicle_id=payload.vehicle_id
    )
    return {"message": "Session initialized successfully", "session_id": payload.session_id}


@app.post("/api/v1/session/end", tags=["Session Lifecycle"])
def end_session(payload: SessionEndRequest):
    """Finalizes an active driving session with aggregated telemetry metrics."""
    db.end_session(
        session_id=payload.session_id,
        total_frames=payload.total_frames,
        avg_ear=payload.avg_ear,
        avg_mar=payload.avg_mar,
        max_fatigue_score=payload.max_fatigue_score
    )
    summary = db.get_session_summary(payload.session_id)
    return {"message": "Session completed", "summary": summary}


@app.get("/api/v1/session/summary/{session_id}", tags=["Session Lifecycle"])
def get_session_summary(session_id: str):
    """Retrieves full post-trip safety telemetry summary."""
    summary = db.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found.")
    return summary


@app.get("/api/v1/monitoring/metrics", tags=["Drift & Performance"])
def get_monitoring_metrics():
    """Returns inference SLA metrics (P50/P95/P99 latency), FPS, and drift indicators."""
    return drift_monitor.compute_drift_indicators()


@app.get("/api/v1/fleet/analytics", tags=["Fleet Analytics"])
def get_fleet_analytics():
    """Aggregates cross-session safety data for enterprise fleet risk profiling."""
    return db.get_fleet_risk_analytics()
