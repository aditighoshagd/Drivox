# Production-grade Container Image for DriverPulse AI
FROM python:3.11-slim

# System dependencies for OpenCV, MediaPipe, and hardware audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies with pip cache disabled for lean container size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source tree
COPY . .

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches FastAPI REST API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
