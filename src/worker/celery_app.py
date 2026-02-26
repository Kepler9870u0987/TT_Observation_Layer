"""Celery application — scheduled tasks for the Observation Layer."""
from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "observation_layer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Scheduled tasks
celery_app.conf.beat_schedule = {
    "nightly-promoter": {
        "task": "src.worker.tasks.run_nightly_promoter",
        "schedule": crontab(hour=2, minute=0),  # ogni notte alle 02:00 UTC
    },
    "daily-retention-cleanup": {
        "task": "src.worker.tasks.apply_retention_policy",
        "schedule": crontab(hour=3, minute=0),  # ogni notte alle 03:00 UTC
    },
}
