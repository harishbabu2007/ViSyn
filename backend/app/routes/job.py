import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..jobs import get_job

router = APIRouter()


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "job_type": job.job_type,
        "error": job.error,
    }

    if job.status == "completed":
        images = {}

        if job.seg_map_path and os.path.exists(job.seg_map_path):
            images["segment_map"] = f"/files/{job.job_id}/input.png"

        if job.job_type == "depth" and job.depth_map_path and os.path.exists(job.depth_map_path):
            images["depth_map"] = f"/files/{job.job_id}/depth.png"

        if job.job_type == "color":
            if job.seg_map_path and os.path.exists(job.seg_map_path):
                images["segment_map"] = f"/files/{job.job_id}/segment.png"
            if job.depth_map_path and os.path.exists(job.depth_map_path):
                images["depth_map"] = f"/files/{job.job_id}/depth.png"
            if job.result_path and os.path.exists(job.result_path):
                images["generated_image"] = f"/files/{job.job_id}/generated.png"

        response["images"] = images

    return JSONResponse(content=response)
