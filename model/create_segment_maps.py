import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation


# CONFIG

# Folder containing input images: 1.png, 2.png, 3.png ...
INPUT_DIR = r"./dataset/inputs"

# Create sibling folder: ../segments
input_path = Path(INPUT_DIR)
OUTPUT_DIR = input_path.parent / "segments"

# Supported image formats
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


# LOAD MODEL

MODEL_NAME = "facebook/maskformer-swin-base-ade"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# CREATE OUTPUT DIRECTORY

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# GET IMAGE FILES
image_files = []

for ext in IMAGE_EXTENSIONS:
    image_files.extend(input_path.glob(f"*{ext}"))

# Sort numerically: 1,2,3...
image_files = sorted(
    image_files,
    key=lambda x: int(x.stem)
)


# PROCESS IMAGES

for image_path in image_files:

    print(f"Processing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    # Convert segmentation map to image
    # Each pixel contains class index
    seg_map = result.cpu().numpy().astype("uint8")

    seg_image = Image.fromarray(seg_map)

    output_path = OUTPUT_DIR / f"{image_path.stem}_segment.png"

    seg_image.save(output_path)

    print(f"Saved: {output_path}")

print("\nDone.")