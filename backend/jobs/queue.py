from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from redis import Redis
from rq import Queue


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def redis_conn() -> Redis:
    return Redis.from_url(_redis_url(), decode_responses=True)


def analysis_queue() -> Queue:
    return Queue("analysis", connection=redis_conn())


def _job_key(job_id: str) -> str:
    return f"analysis_job:{job_id}"


def set_job_status(job_id: str, payload: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    base = {"updated_at": now, **payload}
    redis_conn().set(_job_key(job_id), json.dumps(base), ex=60 * 60 * 24)


def get_job_status(job_id: str) -> dict | None:
    raw = redis_conn().get(_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw)
