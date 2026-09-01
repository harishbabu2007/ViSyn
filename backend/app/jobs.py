import asyncio
import uuid
from typing import Optional

from .config import MAX_CONCURRENT_JOBS
from .models import JobCreate


_semaphore: Optional[asyncio.Semaphore] = None
_jobs: dict[str, JobCreate] = {}


def init_semaphore():
    global _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def get_semaphore() -> asyncio.Semaphore:
    if _semaphore is None:
        init_semaphore()
    return _semaphore


def create_job(job_type: str) -> JobCreate:
    job_id = uuid.uuid4().hex[:12]
    job = JobCreate(job_id=job_id, job_type=job_type)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobCreate]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        for key, value in kwargs.items():
            setattr(_jobs[job_id], key, value)
