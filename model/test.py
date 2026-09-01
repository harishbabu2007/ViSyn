import os
import argparse

import torch

from torchvision.io import read_image
from torchvision.transforms import functional as TF

import torchvision.utils as vutils

from config import *

from generator import Generator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize(x):

    x = x.float() / 255.0

    x = (x * 2.0) - 1.0

    return x


def denormalize(x):

    x = (x + 1.0) / 2.0

    return x.clamp(0, 1)


def load_segmentation(path):

    seg = read_image(path)

    seg = TF.rgb_to_grayscale(
        seg,
        num_output_channels=1
    )

    seg = TF.resize(
        seg,
        [IMG_SIZE, IMG_SIZE]
    )

    seg = normalize(seg)

    seg = seg.unsqueeze(0)

    return seg


def load_generator(
    checkpoint_path,
    condition_channels,
    output_channels,
):

    generator = Generator(
        condition_channels=condition_channels,
        output_channels=output_channels,
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "generator" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator"]
        )

    else:

        generator.load_state_dict(checkpoint)

    generator.eval()

    return generator


@torch.no_grad()
def run_pipeline(
    seg_path,
    s2d_checkpoint,
    sd2i_checkpoint,
    output_image_path,
    output_depth_path=None,
):

    seg = load_segmentation(seg_path).to(DEVICE)

    s2d_generator = load_generator(
        s2d_checkpoint,
        condition_channels=1,
        output_channels=1,
    )

    fake_depth = s2d_generator(seg)

    sd2i_generator = load_generator(
        sd2i_checkpoint,
        condition_channels=2,
        output_channels=3,
    )

    condition = torch.cat(
        [seg, fake_depth],
        dim=1
    )

    fake_image = sd2i_generator(condition)

    fake_image = denormalize(fake_image)

    os.makedirs(
        os.path.dirname(output_image_path),
        exist_ok=True
    )

    vutils.save_image(
        fake_image,
        output_image_path
    )

    print(f"saved image -> {output_image_path}")

    if output_depth_path is not None:

        fake_depth_vis = denormalize(fake_depth)

        os.makedirs(
            os.path.dirname(output_depth_path),
            exist_ok=True
        )

        vutils.save_image(
            fake_depth_vis,
            output_depth_path
        )

        print(f"saved depth -> {output_depth_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seg",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--s2d",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sd2i",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-image",
        type=str,
        default="outputs/generated.png",
    )

    parser.add_argument(
        "--output-depth",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    run_pipeline(
        seg_path=args.seg,
        s2d_checkpoint=args.s2d,
        sd2i_checkpoint=args.sd2i,
        output_image_path=args.output_image,
        output_depth_path=args.output_depth,
    )


if __name__ == "__main__":
    main()