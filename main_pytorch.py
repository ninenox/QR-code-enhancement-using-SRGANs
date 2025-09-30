"""PyTorch implementation of SRGAN components for QR code enhancement."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class ResidualBlock(nn.Module):
    """Residual block used inside the generator."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBlock(nn.Module):
    """Upsample block mirroring the original Keras implementation."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.upsample(x)
        return self.prelu(x)


class Generator(nn.Module):
    """SRGAN generator network."""

    def __init__(self, res_blocks: int = 1, upsample_blocks: int = 1) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual_layers = nn.Sequential(*[ResidualBlock(64) for _ in range(res_blocks)])

        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        upsample_layers = []
        in_channels = 64
        for _ in range(upsample_blocks):
            upsample_layers.append(UpsampleBlock(in_channels))
            in_channels = 256
        self.upsample_layers = nn.Sequential(*upsample_layers)

        self.output = nn.Conv2d(in_channels, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial = self.initial(x)
        out = self.residual_layers(initial)
        out = self.conv_block(out)
        out = out + initial
        out = self.upsample_layers(out)
        return self.output(out)


class DiscriminatorBlock(nn.Module):
    """Single block for the discriminator network."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """SRGAN discriminator network."""

    def __init__(self) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DiscriminatorBlock(64, 64, stride=2),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 128, stride=2),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 256, stride=2),
            DiscriminatorBlock(256, 512),
            DiscriminatorBlock(512, 512, stride=2)
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return self.classifier(out)


def build_vgg(hr_shape: Optional[Tuple[int, int, int]] = None) -> nn.Module:
    """Return a truncated VGG19 model matching the Keras implementation."""

    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    features = list(model.features.children())[:10]
    vgg = nn.Sequential(*features)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


class CombinedModel(nn.Module):
    """Combined model for adversarial and perceptual training."""

    def __init__(self, generator: Generator, discriminator: Discriminator, vgg: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        for param in self.discriminator.parameters():
            param.requires_grad = False
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, lr: torch.Tensor, hr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        gen_img = self.generator(lr)
        features = self.vgg(gen_img)
        validity = self.discriminator(gen_img)
        return validity, features


def create_comb(generator: Generator, discriminator: Discriminator, vgg: nn.Module) -> CombinedModel:
    """Factory method mirroring the Keras helper."""

    return CombinedModel(generator, discriminator, vgg)


__all__ = [
    "ResidualBlock",
    "UpsampleBlock",
    "Generator",
    "Discriminator",
    "build_vgg",
    "create_comb",
    "CombinedModel",
]
