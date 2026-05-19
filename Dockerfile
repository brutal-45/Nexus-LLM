# ============================================================
# Nexus-LLM Dockerfile - Multi-stage Build
# ============================================================

# ----------------------------------------------------------
# Stage 1: Builder - Install dependencies
# ----------------------------------------------------------
FROM python:3.11-slim AS builder

ARG CUDA_VERSION=12.1
ARG TORCH_VERSION=2.1.0
ARG INSTALL_GPU=false

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Conditionally install GPU dependencies
RUN if [ "$INSTALL_GPU" = "true" ]; then \
    pip install --no-cache-dir -r requirements-gpu.txt || true; \
    fi

# Copy the project
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .


# ----------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ----------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG INSTALL_GPU=false

LABEL maintainer="Nexus-LLM Team <nexus-llm@example.com>"
LABEL description="Nexus-LLM: A powerful LLM framework for training, serving, and chatting"
LABEL version="0.1.0"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Conditionally install CUDA runtime libraries
RUN if [ "$INSTALL_GPU" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-1 \
    cuda-runtime-12-1 \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Create non-root user
RUN groupadd -r nexus && useradd -r -g nexus -d /home/nexus -s /sbin/nologin nexus

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/cache /app/logs /app/output && \
    chown -R nexus:nexus /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=nexus:nexus . /app

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CACHE_DIR=/app/cache \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    NEXUS_LLM_HOST=0.0.0.0 \
    NEXUS_LLM_PORT=8000 \
    NEXUS_LLM_LOG_LEVEL=INFO

# Switch to non-root user
USER nexus

# Expose the server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entry point
ENTRYPOINT ["python", "main.py"]

# Default command
CMD ["--help"]
