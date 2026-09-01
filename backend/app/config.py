import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR.parent / "model"

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

S2D_CHECKPOINT = CHECKPOINT_DIR / "s2d_latest.pth"
SD2I_CHECKPOINT = CHECKPOINT_DIR / "sd2i_latest.pth"

UPLOAD_DIR = Path("/tmp") / "vsyn"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

IMG_SIZE = 256

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    os.getenv("FRONTEND_URL", ""),
]
