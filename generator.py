import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from config import *


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()       
        self.eps = eps

    def forward(self, x):
        return x / (1 / x.size(1) * (x ** 2).sum(dim=1, keepdim=True) + self.eps).sqrt()


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, c0, h0, w0, n_layers):
        super().__init__()
        layers = []
        in_dim = z_dim

        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, 256), PixelNorm(), nn.LeakyReLU(0.2)]
            in_dim = 256
        layers += [nn.Linear(256, c0 * h0 * w0), PixelNorm(), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)
        self.c0, self.h0, self.w0 = c0, h0, w0

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], self.c0, self.h0, self.w0)
        return x


class ConditionPreparationModule(nn.Module):
    def __init__(self, target_sizes, out_channels, in_channels=3):
        super().__init__()

        self.target_sizes = target_sizes
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channel, 3, 1, 1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            for out_channel in out_channels
        ])

    def forward(self, x):
        outputs = []
        for size, conv in zip(self.target_sizes, self.conv_layers):
            x_resized = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            outputs.append(conv(x_resized))
        return outputs


class ConditionFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
            for ch in channels
        ])

        self.upsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1),
                nn.ReLU(inplace=True)
            )
            for i in range(len(channels) - 1)
        ])

    def forward(self, w, outputs):
        x = w
        w_plus_list = []

        for i, (conv, mi) in enumerate(zip(self.convs, outputs)):
            x = x + mi
            x = conv(x)
            w_plus_list.append(x)

            if i < len(self.upsample_convs):
                next_size = outputs[i + 1].shape[-2:]
                x = F.interpolate(x, size=next_size, mode="bilinear", align_corners=False)
                x = self.upsample_convs[i](x)

        return w_plus_list


class AffineTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels + out_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, w_plus):
        w_flat = w_plus.mean(dim=[2, 3])          # (B, in_ch)
        style = self.linear(w_flat)               # (B, in_ch + out_ch)
        gamma = style[:, :self.in_channels]       # (B, in_ch)
        beta = style[:, self.in_channels:]        # (B, out_ch)
        return gamma, beta


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x, gamma, beta):
        B, C, H, W = x.shape

        w = self.weight.unsqueeze(0)              # (1, out, in, kH, kW)
        gamma = gamma.view(B, 1, C, 1, 1)
        w_mod = w * (gamma + 1)                   # (B, out, in, kH, kW)

        sigma = (w_mod.pow(2).sum(dim=[2, 3, 4]) + 1e-8).sqrt()
        w_demod = w_mod / sigma.view(B, -1, 1, 1, 1)

        x_in = x.contiguous().view(1, B * C, H, W)
        w_in = w_demod.contiguous().view(
            B * w_demod.shape[1], C, self.weight.shape[2], self.weight.shape[3]
        )
        out = F.conv2d(x_in, w_in, padding=self.padding, groups=B)
        out = out.view(B, -1, H, W)

        out = out + beta.view(B, -1, 1, 1)
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.affine = AffineTransform(in_ch, out_ch)
        self.mod_conv = ModulatedConv2d(in_ch, out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, w_plus):
        # 1. Affine: w+ → gamma, beta
        gamma, beta = self.affine(w_plus)

        # 2. Modulated conv
        x = self.mod_conv(x, gamma, beta)

        # 3. Regular conv
        x = self.conv(x)

        x = F.relu(x)

        # 4. Add noise (B block)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_scale * noise

        return x


class ImageSynthesisModule(nn.Module):
    def __init__(self, channels, C0, H0, W0):
        super().__init__()

        self.const = nn.Parameter(torch.randn(1, C0, H0, W0))

        self.blocks = nn.ModuleList([
            SynthesisBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(channels, channels[1:])
        ])

        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, w_plus_list):
        B = w_plus_list[0].shape[0]
        x = self.const.repeat(B, 1, 1, 1)

        for block, w_plus in zip(self.blocks, w_plus_list):
            if x.shape[-2:] != w_plus.shape[-2:]:
                x = F.interpolate(x, size=w_plus.shape[-2:], mode="bilinear", align_corners=False)
            x = block(x, w_plus)

        return self.to_rgb(x)


class Generator(nn.Module):
    def __init__(self, target_sizes=target_sizes, out_channels=out_channels, w0=w0, h0=h0):
        super().__init__()      
        self.target_sizes = target_sizes
        self.out_channels = out_channels    
        self.w0 = w0                        
        self.h0 = h0

        self.mapping_network = MappingNetwork(
            64, self.out_channels[0], self.h0, self.w0, 4
        )
        self.condition_preparation = ConditionPreparationModule(
            self.target_sizes, self.out_channels, 3
        )
        self.condition_fusion = ConditionFusionModule(self.out_channels)
        self.image_synthesis = ImageSynthesisModule(
            self.out_channels, self.out_channels[0], self.h0, self.w0
        )

    def forward(self, condition_map):
        batch_size = condition_map.size(0)
        z = torch.randn(batch_size, 64, device=condition_map.device)
        w = self.mapping_network(z)
        outputs = self.condition_preparation(condition_map)
        w_plus_list = self.condition_fusion(w, outputs)
        image = self.image_synthesis(w_plus_list)
        return image
