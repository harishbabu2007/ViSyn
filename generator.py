import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8
        )


class MappingNetwork(nn.Module):

    def __init__(self, z_dim=64, w_dim=512, n_layers=8):

        super().__init__()

        layers = []

        for _ in range(n_layers):

            layers.append(
                nn.Linear(z_dim if len(layers) == 0 else w_dim, w_dim)
            )

            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):

        z = PixelNorm()(z)

        w = self.mapping(z)

        return w



class ConditionPreparationModule(nn.Module):

    def __init__(
        self,
        condition_channels,
        target_sizes,
        channels,
    ):

        super().__init__()

        self.blocks = nn.ModuleList()

        for ch in channels:

            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(condition_channels, ch, 3, 1, 1),
                    nn.InstanceNorm2d(ch),
                    nn.LeakyReLU(0.2),
                )
            )

        self.target_sizes = target_sizes

    def forward(self, x):

        outputs = []

        for size, block in zip(self.target_sizes, self.blocks):

            resized = F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

            out = block(resized)

            outputs.append(out)

        return outputs


class NoiseInjection(nn.Module):

    def __init__(self):

        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, _, H, W = x.shape

        noise = torch.randn(
            B,
            1,
            H,
            W,
            device=x.device
        )

        return x + self.weight * noise


class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):

        super().__init__()

        self.norm = nn.InstanceNorm2d(channels)

        self.style = nn.Linear(w_dim, channels * 2)

    def forward(self, x, w):

        style = self.style(w)

        gamma, beta = style.chunk(2, dim=1)

        gamma = gamma.unsqueeze(2).unsqueeze(3)

        beta = beta.unsqueeze(2).unsqueeze(3)

        x = self.norm(x)

        x = gamma * x + beta

        return x


class StyledConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            1,
            1
        )

        self.noise1 = NoiseInjection()

        self.adain1 = AdaIN(out_channels, w_dim)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1
        )

        self.noise2 = NoiseInjection()

        self.adain2 = AdaIN(out_channels, w_dim)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):

        x = self.conv1(x)

        x = self.noise1(x)

        x = self.activation(x)

        x = self.adain1(x, w)

        x = self.conv2(x)

        x = self.noise2(x)

        x = self.activation(x)

        x = self.adain2(x, w)

        return x


class ToRGB(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            0
        )

    def forward(self, x):

        return self.conv(x)



class Generator(nn.Module):

    def __init__(
        self,
        condition_channels,
        output_channels,
        z_dim=Z_DIM,
        w_dim=512,
        target_sizes=TARGET_SIZES,
        channels=CHANNELS,
    ):

        super().__init__()

        self.target_sizes = target_sizes

        self.channels = channels

        self.mapping_network = MappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
        )

        self.condition_preparation = ConditionPreparationModule(
            condition_channels=condition_channels,
            target_sizes=target_sizes,
            channels=channels,
        )

        self.constant = nn.Parameter(
            torch.randn(
                1,
                channels[0],
                W0,
                H0
            )
        )

        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):

            self.blocks.append(
                StyledConvBlock(
                    channels[i],
                    channels[i + 1],
                    w_dim,
                )
            )

        self.to_rgb = ToRGB(
            channels[-1],
            output_channels,
        )

    def forward(self, condition_map):

        B = condition_map.shape[0]


        z = torch.randn(
            B,
            Z_DIM,
            device=condition_map.device
        )

        w = self.mapping_network(z)


        cond_features = self.condition_preparation(
            condition_map
        )


        x = self.constant.repeat(B, 1, 1, 1)

        # fuse first condition
        x = x + cond_features[0]


        for i, block in enumerate(self.blocks):

            next_size = self.target_sizes[i + 1]

            x = F.interpolate(
                x,
                size=next_size,
                mode="bilinear",
                align_corners=False,
            )

            # condition fusion
            x = x + cond_features[i + 1]

            x = block(x, w)


        out = self.to_rgb(x)

        out = torch.tanh(out)

        return out



if __name__ == "__main__":

    # SD2I
    x = torch.randn(2, 4, 256, 256)

    gen = Generator(
        condition_channels=4,
        output_channels=3,
    )

    y = gen(x)

    print("SD2I:", y.shape)

    # S2D
    x2 = torch.randn(2, 3, 256, 256)

    gen2 = Generator(
        condition_channels=3,
        output_channels=1,
    )

    y2 = gen2(x2)

    print("S2D:", y2.shape)