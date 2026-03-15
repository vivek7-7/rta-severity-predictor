# Dockerfile
# Multi-stage build for the RTA Severity Predictor FastAPI application.
# Base image: python:3.11-slim

FROM python:3.11-slim AS base

# System deps needed by LightGBM and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency layer (cached unless requirements.txt changes) ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# Create artifacts directory (will be populated at runtime via volume or training)
RUN mkdir -p app/ml/artifacts

# Expose the application port
EXPOSE 8000

# ── Runtime command ────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
