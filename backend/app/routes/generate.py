import asyncio
import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..config import UPLOAD_DIR, MAX_CONCURRENT_JOBS
from ..jobs import create_job, get_semaphore, update_job
from ..inference import generate_depth, generate_color

router = APIRouter()


async def _run_depth_job(job_id: str, seg_path: str, output_dir: str):
    sem = get_semaphore()
    async with sem:
        try:
            output_path = f"{output_dir}/depth.png"
            await asyncio.to_thread(generate_depth, seg_path, output_path)
            update_job(
                job_id,
                status="completed",
                depth_map_path=output_path,
            )
        except Exception as e:
            traceback.print_exc()
            update_job(job_id, status="failed", error=str(e))


async def _run_color_job(job_id: str, seg_path: str, depth_path: str, output_dir: str):
    sem = get_semaphore()
    async with sem:
        try:
            output_path = f"{output_dir}/generated.png"
            await asyncio.to_thread(generate_color, seg_path, depth_path, output_path)
            update_job(
                job_id,
                status="completed",
                result_path=output_path,
            )
        except Exception as e:
            traceback.print_exc()
            update_job(job_id, status="failed", error=str(e))


@router.post("/generate-depth")
async def create_depth_job(segment_map: UploadFile = File(...)):
    sem = get_semaphore()
    if sem.locked():
        raise HTTPException(
            status_code=429,
            detail="Server overloaded. Try again later.",
        )

    job = create_job(job_type="depth")
    job_dir = str(UPLOAD_DIR / job.job_id)
    import os
    os.makedirs(job_dir, exist_ok=True)

    seg_path = f"{job_dir}/input.png"
    content = await segment_map.read()
    with open(seg_path, "wb") as f:
        f.write(content)

    update_job(job.job_id, seg_map_path=seg_path)

    asyncio.create_task(_run_depth_job(job.job_id, seg_path, job_dir))

    return JSONResponse(content={"job_id": job.job_id})


@router.post("/generate-color")
async def create_color_job(
    segment_map: UploadFile = File(...),
    depth_map: UploadFile = File(...),
):
    sem = get_semaphore()
    if sem.locked():
        raise HTTPException(
            status_code=429,
            detail="Server overloaded. Try again later.",
        )

    job = create_job(job_type="color")
    job_dir = str(UPLOAD_DIR / job.job_id)
    import os
    os.makedirs(job_dir, exist_ok=True)

    seg_path = f"{job_dir}/segment.png"
    depth_path = f"{job_dir}/depth.png"

    seg_content = await segment_map.read()
    with open(seg_path, "wb") as f:
        f.write(seg_content)

    depth_content = await depth_map.read()
    with open(depth_path, "wb") as f:
        f.write(depth_content)

    update_job(
        job.job_id,
        seg_map_path=seg_path,
        depth_map_path=depth_path,
    )

    asyncio.create_task(_run_color_job(job.job_id, seg_path, depth_path, job_dir))

    return JSONResponse(content={"job_id": job.job_id})
