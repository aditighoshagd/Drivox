"""
Drivox - Main System Entry Point
Production-Ready Edge & Cloud Driver Safety Intelligence System.
Supports multiple operating modes:
  - dashboard: Interactive Streamlit analytics & live video interface
  - api: FastAPI REST microservice for edge frame processing and telemetry
  - cli: Headless / OpenCV live camera monitoring
  - test: Automated test verification suite
"""

import sys
import os
import argparse
import subprocess


def run_dashboard():
    """Launches the Streamlit interactive dashboard."""
    print("[INFO] Launching Drivox Dashboard...")
    dashboard_path = os.path.join(os.path.dirname(__file__), "src", "dashboard", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Launches the FastAPI REST microservice."""
    print(f"[INFO] Starting Drivox REST API at http://{host}:{port} ...")
    print(f"[INFO] Interactive OpenAPI / Swagger documentation: http://{host}:{port}/docs")
    try:
        import uvicorn
        uvicorn.run("src.api.app:app", host=host, port=port, reload=True)
    except ImportError:
        print("[ERROR] uvicorn is not installed. Run: pip install uvicorn fastapi")


def run_cli_stream():
    """Runs real-time OpenCV edge monitoring on webcam."""
    print("[INFO] Starting Edge Vision Stream. Press 'q' to exit.")
    try:
        import cv2
        from src.core.detector import FatigueDetector
        from src.core.alert_manager import AlertManager
        from src.pipeline.data_pipeline import ImagePreprocessingPipeline
        from src.monitoring.drift_monitor import DriftMonitor

        detector = FatigueDetector()
        alert_mgr = AlertManager()
        pipeline = ImagePreprocessingPipeline()
        drift_monitor = DriftMonitor()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open video device 0.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = pipeline.preprocess_stream_frame(frame)
            result, landmarks = detector.process_frame(processed)
            alert_mgr.evaluate(result.status, result.fatigue_score)

            # Overlay telemetry
            color = (0, 255, 0)
            if result.status == "DROWSY":
                color = (0, 0, 255)
            elif result.status in ("DISTRACTED", "YAWN_WARNING"):
                color = (0, 165, 255)

            cv2.putText(frame, f"STATUS: {result.status}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {result.ear:.2f} | MAR: {result.mar:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Fatigue Score: {result.fatigue_score:.2f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Drivox - Live Edge Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] Edge stream failure: {e}")


def run_tests():
    """Runs system test suite (supports both pytest and unittest)."""
    print("[INFO] Executing automated test suite...")
    try:
        import pytest
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    except ImportError:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "tests", "-v"])


def main():
    parser = argparse.ArgumentParser(description="Drivox - Driver Fatigue & Safety Intelligence")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "api", "cli", "test"],
        default="dashboard",
        help="System runtime mode (default: dashboard)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API bind host")
    parser.add_argument("--port", type=int, default=8000, help="API bind port")

    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "api":
        run_api(host=args.host, port=args.port)
    elif args.mode == "cli":
        run_cli_stream()
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()