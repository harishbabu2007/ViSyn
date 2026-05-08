import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
    ):

        super().__init__()

        self.block = nn.Sequential(

            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                )
            ),

            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):

        return self.block(x)


class Discriminator(nn.Module):

    def __init__(
        self,
        in_channels,
        base_channels=64,
    ):

        super().__init__()

        self.model = nn.Sequential(

            # 256 -> 128
            DiscBlock(
                in_channels,
                base_channels,
            ),

            # 128 -> 64
            DiscBlock(
                base_channels,
                base_channels * 2,
            ),

            # 64 -> 32
            DiscBlock(
                base_channels * 2,
                base_channels * 4,
            ),

            # 32 -> 16
            DiscBlock(
                base_channels * 4,
                base_channels * 8,
            ),

            # 16 -> 15
            spectral_norm(
                nn.Conv2d(
                    base_channels * 8,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                )
            )
        )

    def forward(self, x):

        return self.model(x)


if __name__ == "__main__":

    # SD2I
    # seg = 3
    # depth = 1
    # image = 3
    # total = 7


    x = torch.randn(2, 7, 256, 256)

    disc = Discriminator(
        in_channels=7
    )

    y = disc(x)

    print("SD2I:", y.shape)

    # S2D
    # seg = 3
    # depth = 1
    # total = 4

    x2 = torch.randn(2, 4, 256, 256)

    disc2 = Discriminator(
        in_channels=4
    )

    y2 = disc2(x2)

    print("S2D:", y2.shape)