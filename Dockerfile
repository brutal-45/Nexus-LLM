# syntax=docker/dockerfile:1
# ============================================================================
# Nexus-LLM container image
#
#   CPU (default):
#     docker build -t nexus-llm .
#   GPU (CUDA 12.1):
#     docker build --build-arg CUDA_VERSION=12.1 -t nexus-llm:gpu .
#     docker run --gpus all -p 8000:8000 nexus-llm:gpu
#
# The project is installed non-editable from pyproject.toml, so the image
# contains exactly the published package (including its bundled config,
# presets, templates and i18n data files).
# ============================================================================

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

ARG CUDA_VERSION=""
# "" selects the CPU wheels from PyPI's torch index; "12.1" etc. for GPU.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /src

# torch must come from the PyTorch index so the right CUDA/CPU build is chosen.
COPY requirements.txt ./
RUN set -eux; \
    python -m pip install --upgrade pip setuptools wheel; \
    if [ -n "${CUDA_VERSION}" ]; then \
      python -m pip install --index-url "${TORCH_INDEX_URL}" "torch" "torchvision"; \
    else \
      python -m pip install --index-url "${TORCH_INDEX_URL}" "torch"; \
    fi; \
    python -m pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

# Package metadata first, so dependency resolution is cached across code edits.
COPY pyproject.toml README.md LICENSE VERSION ./
COPY nexus_llm ./nexus_llm
COPY config ./config
RUN python -m pip install --no-deps .


# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG PYTHON_VERSION
# Derived from PYTHON_VERSION so the copied site-packages can never diverge
# from the base image tag.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages

LABEL org.opencontainers.image.title="Nexus-LLM" \
      org.opencontainers.image.description="Terminal LLM chat with a local inference backend" \
      org.opencontainers.image.source="https://github.com/brutal-45/Nexus-LLM" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEXUS_HOME=/var/lib/nexus-llm \
    XDG_CACHE_HOME=/var/cache/nexus-llm \
    NEXUS_CACHE_DIR=/var/cache/nexus-llm \
    HF_HOME=/var/cache/nexus-llm/huggingface \
    NEXUS_CONFIG=/app/config/default_config.yaml \
    TOKENIZERS_PARALLELISM=false

# tini reaps zombie processes and forwards signals, so uvicorn shuts down
# cleanly on `docker stop`. curl is used by the healthcheck.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends tini curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --create-home --shell /usr/sbin/nologin --uid 10001 nexus; \
    mkdir -p /var/lib/nexus-llm /var/cache/nexus-llm; \
    chown -R nexus:nexus /var/lib/nexus-llm /var/cache/nexus-llm

COPY --from=builder ${SITE_PACKAGES} ${SITE_PACKAGES}
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
# Only the editable config ships alongside the installed package; the library
# itself (presets, templates, i18n data) already lives in site-packages.
COPY --chown=nexus:nexus config ./config

USER nexus
EXPOSE 8000
VOLUME ["/var/lib/nexus-llm", "/var/cache/nexus-llm"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["tini", "--", "nexus-llm"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
