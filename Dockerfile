# =========================================
# Detectify Dockerfile - Multi-stage build
# =========================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Export requirements
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =========================================
# Stage 2: Runtime (GPU-enabled)
# =========================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install Python and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Environment variables
ENV DATABASE_URL=sqlite:///detectify.db
ENV TFHUB_CACHE_DIR=/app/model_cache
ENV PYTHONPATH=/app/src

# Create cache directory
RUN mkdir -p /app/model_cache

# Expose port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "detectify.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# =========================================
# Stage 3: Runtime (CPU-only, smaller)
# =========================================
FROM python:3.11-slim AS runtime-cpu

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Environment variables
ENV DATABASE_URL=sqlite:///detectify.db
ENV TFHUB_CACHE_DIR=/app/model_cache
ENV PYTHONPATH=/app/src

# Create cache directory
RUN mkdir -p /app/model_cache

# Expose port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "detectify.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
