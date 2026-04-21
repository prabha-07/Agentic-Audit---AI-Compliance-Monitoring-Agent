# syntax=docker/dockerfile:1.6
# --------------------------------------------------------------------------- #
# Build stage — bundle the React frontend
# --------------------------------------------------------------------------- #
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
# Runtime stage — CPU-only Python for small-RAM hosts (Render free tier)
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    # Keep transformers/torch quiet about CUDA in a CPU-only container.
    CUDA_VISIBLE_DEVICES="" \
    TOKENIZERS_PARALLELISM=false

# build-essential is only needed for a few wheels; strip it after install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch wheel first (≈ 200 MB vs. the default 2 GB CUDA wheel).
# This is critical for Render free tier disk + RAM budgets.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Strip build toolchain after wheels are installed to shrink the image.
RUN apt-get purge -y --auto-remove build-essential

COPY backend ./backend
COPY run_pipeline.py run_evaluation.py ./
COPY data ./data
COPY scripts ./scripts
COPY --from=ui /ui/dist ./frontend/dist

# The regulation index is built at first startup (or pre-baked into a Render
# persistent disk). Doing it here would need ~400 MB RAM during build and
# HF downloads, which often flake on free builders.
RUN mkdir -p /app/.cache/huggingface /app/data/chroma_db /app/outputs/reports /app/outputs/logs

EXPOSE 8000

# Render injects $PORT; uvicorn binds to it.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
