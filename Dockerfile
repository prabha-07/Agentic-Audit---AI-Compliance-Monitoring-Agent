# --- Build Web UI (Node) ---
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Runtime (Python) ---
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend ./backend
COPY run_pipeline.py ./
COPY data ./data
COPY scripts ./scripts

COPY --from=ui /ui/dist ./frontend/dist

# Persisted Chroma index (not in git). Pre-build embeddings so the API can retrieve at runtime.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface /app/data/chroma_db \
    && python scripts/index_regulations.py --all

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
