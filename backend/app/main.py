import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, UPLOAD_DIR
from .jobs import init_semaphore
from .inference import load_models
from .routes.generate import router as generate_router
from .routes.job import router as job_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_semaphore()
    load_models()
    yield


app = FastAPI(
    title="ViSyn API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in CORS_ORIGINS if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router, prefix="/api")
app.include_router(job_router, prefix="/api")

if os.path.exists(UPLOAD_DIR):
    app.mount(
        "/files",
        StaticFiles(directory=str(UPLOAD_DIR)),
        name="files",
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
