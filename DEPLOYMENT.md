# 🚀 Deployment Guide - Drivox

This document provides instructions for deploying **Drivox** across edge devices, on-prem servers, and cloud container environments.

---

## 1. Quick Start (Local Run)

### Prerequisites
* Python 3.10 or 3.11
* Web camera (for edge video streams)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Dashboard
python main.py --mode dashboard
```

---

## 2. Microservice Deployment (FastAPI + Streamlit via Docker)

The recommended production deployment utilizes Docker Compose to orchestrate both the REST API microservice and the Streamlit analytics interface:

```bash
# Build and run containers
docker compose -f docker/docker-compose.yml up --build -d
```

### Services Deployed
* **REST API**: `http://localhost:8000` (OpenAPI Swagger docs at `http://localhost:8000/docs`)
* **Dashboard**: `http://localhost:8501`

To view container logs:
```bash
docker compose -f docker/docker-compose.yml logs -f
```

To stop containers:
```bash
docker compose -f docker/docker-compose.yml down
```

---

## 3. Cloud Deployment (AWS / GCP / Azure VM or ECS)

### Systemd Service Setup (Linux Server / In-Cabin Computer)
Create `/etc/systemd/system/drivox-api.service`:

```ini
[Unit]
Description=Drivox REST Microservice
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/Drivox
ExecStart=/opt/Drivox/venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable drivox-api
sudo systemctl start drivox-api
```
