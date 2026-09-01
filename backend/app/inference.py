import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from .config import MODEL_DIR, CHECKPOINT_DIR, IMG_SIZE


_s2d_model = None
_sd2i_model = None


def _load_generator(checkpoint_path: Path, condition_channels: int, output_channels: int):
    sys.path.insert(0, str(MODEL_DIR))

    from generator import Generator

    generator = Generator(
        condition_channels=condition_channels,
        output_channels=output_channels,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("generator", checkpoint)
    generator.load_state_dict(state_dict)
    generator.eval()

    sys.path.pop(0)

    return generator


def load_models():
    global _s2d_model, _sd2i_model

    s2d_path = CHECKPOINT_DIR / "s2d_latest.pth"
    sd2i_path = CHECKPOINT_DIR / "sd2i_latest.pth"

    if not s2d_path.exists() or not sd2i_path.exists():
        print("[inference] Checkpoint files not found. Models not loaded.")
        print(f"[inference] Expected: {s2d_path}")
        print(f"[inference] Expected: {sd2i_path}")
        return

    print("[inference] Loading S2D model...")
    _s2d_model = _load_generator(
        checkpoint_path=s2d_path,
        condition_channels=1,
        output_channels=1,
    )

    print("[inference] Loading SD2I model...")
    _sd2i_model = _load_generator(
        checkpoint_path=sd2i_path,
        condition_channels=2,
        output_channels=3,
    )

    print("[inference] Both models loaded.")


def _preprocess(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    arr = (arr * 2.0) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def _postprocess(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return arr


@torch.no_grad()
def generate_depth(seg_image_path: str, output_path: str):
    global _s2d_model

    if _s2d_model is None:
        raise RuntimeError("S2D model not loaded")

    seg_tensor = _preprocess(seg_image_path)
    depth_tensor = _s2d_model(seg_tensor)
    depth_np = _postprocess(depth_tensor)

    if depth_np.ndim == 3 and depth_np.shape[2] == 1:
        depth_np = depth_np.squeeze(2)

    img = Image.fromarray(depth_np, mode="L")
    img.save(output_path)
    return output_path


@torch.no_grad()
def generate_color(seg_image_path: str, depth_image_path: str, output_path: str):
    global _sd2i_model

    if _sd2i_model is None:
        raise RuntimeError("SD2I model not loaded")

    seg_tensor = _preprocess(seg_image_path)
    depth_tensor = _preprocess(depth_image_path)
    condition = torch.cat([seg_tensor, depth_tensor], dim=1)
    color_tensor = _sd2i_model(condition)
    color_np = _postprocess(color_tensor)

    img = Image.fromarray(color_np, mode="RGB")
    img.save(output_path)
    return output_path
