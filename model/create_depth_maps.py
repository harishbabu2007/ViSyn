import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import pipeline


# CONFIG

# Input image folder
INPUT_DIR = r"./dataset/inputs"

input_path = Path(INPUT_DIR)

# Output folder (sibling directory)
OUTPUT_DIR = input_path.parent / "depths"

# Supported formats
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


# DEVICE

device = 0 if torch.cuda.is_available() else -1


# LOAD DEPTH MODEL
# Uses HuggingFace depth-estimation pipeline
depth_estimator = pipeline(
    task="depth-estimation",
    model="Intel/dpt-large",
    device=device
)

# CREATE OUTPUT DIRECTORY

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# GET IMAGE FILES

image_files = []

for ext in IMAGE_EXTENSIONS:
    image_files.extend(input_path.glob(f"*{ext}"))

# Sort numerically
image_files = sorted(
    image_files,
    key=lambda x: int(x.stem)
)


# PROCESS IMAGES

for image_path in image_files:

    print(f"Processing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")

    # Run depth estimation
    result = depth_estimator(image)

    depth = result["depth"]

    # Convert PIL image -> numpy
    depth_np = np.array(depth)

    # Normalize to 0-255
    depth_np = depth_np.astype(np.float32)

    depth_np = (
        (depth_np - depth_np.min()) /
        (depth_np.max() - depth_np.min())
    ) * 255.0

    depth_np = depth_np.astype(np.uint8)

    # Save grayscale depth map
    depth_image = Image.fromarray(depth_np)

    output_path = OUTPUT_DIR / f"{image_path.stem}_depth.png"

    depth_image.save(output_path)

    print(f"Saved: {output_path}")

print("\nDone.")