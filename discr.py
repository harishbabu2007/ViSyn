import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, base_channels=64):
        super().__init__()
        C = base_channels
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, C,     kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C,     C * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(C * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C * 2, C * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(C * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C * 4, C * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(C * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(C * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.disc(x)

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    disc = Discriminator()
    print(disc(x).shape)
