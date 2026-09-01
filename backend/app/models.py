from pydantic import BaseModel
from typing import Optional


class JobCreate(BaseModel):
    job_id: str
    job_type: str  # "depth" or "color"
    status: str = "processing"
    error: Optional[str] = None
    seg_map_path: Optional[str] = None
    depth_map_path: Optional[str] = None
    result_path: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    error: Optional[str] = None
    images: Optional[dict] = None
