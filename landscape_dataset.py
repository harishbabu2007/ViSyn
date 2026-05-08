import os
import torch
from torchvision.io import decode_image
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset


class LandscapesDataset(Dataset):
    def __init__(
        self,
        num_images,
        input_dir,
        segment_dir,
        depth_dir,
        image_size=(256, 256),
    ):
        self.input_dir = input_dir
        self.segment_dir = segment_dir
        self.depth_dir = depth_dir
        self.num_images = num_images
        self.image_size = image_size

    def __len__(self):
        return self.num_images

    def normalize(self, x):
        x = x.float() / 255.0
        x = (x * 2.0) - 1.0
        return x

    def __getitem__(self, idx):

        truth_name = f"{idx}.jpeg"
        seg_name = f"{idx}_segment.png"
        depth_name = f"{idx}_depth.png"

        truth_path = os.path.join(self.input_dir, truth_name)
        seg_path = os.path.join(self.segment_dir, seg_name)
        depth_path = os.path.join(self.depth_dir, depth_name)

        truth = decode_image(truth_path)
        seg = decode_image(seg_path)
        depth = decode_image(depth_path)

        # force grayscale depth
        depth = TF.rgb_to_grayscale(depth, num_output_channels=1)

        # resize
        truth = TF.resize(truth, self.image_size)
        seg = TF.resize(seg, self.image_size)
        depth = TF.resize(depth, self.image_size)

        # normalize to [-1, 1]
        truth = self.normalize(truth)
        seg = self.normalize(seg)
        depth = self.normalize(depth)

        return {
            "image": truth,
            "seg": seg,
            "depth": depth,
        }