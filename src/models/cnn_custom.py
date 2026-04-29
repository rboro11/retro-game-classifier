"""
Phase 1 / Phase 2 Model — Custom Lightweight CNN

Good for NES-era games (low-res, pixel-art style).
Deliberately small so it trains fast on CPU or free Colab GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MarioCNNSmall(nn.Module):
    """
    3-block CNN for ~64×64 or ~128×128 input.
    Best for Phase 1 (binary) and Phase 2 (3–5 classes, NES era).

    Architecture:
        Input: (B, 3, img_size, img_size)
        ConvBlock(3→32, pool)   → (B, 32, img/2, img/2)
        ConvBlock(32→64, pool)  → (B, 64, img/4, img/4)
        ConvBlock(64→128, pool) → (B, 128, img/8, img/8)
        GlobalAvgPool           → (B, 128)
        Dropout(0.5)
        Linear → num_classes
    """

    def __init__(self, num_classes: int, img_size: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MarioCNNMedium(nn.Module):
    """
    5-block CNN for ~224×224 input.
    Suitable for Phase 2–3 when not using pretrained transfer learning.

    Architecture:
        4 ConvBlocks with progressive channel expansion
        Residual skip every 2 blocks (mini-ResNet style)
        Global Average Pooling → Dropout → Linear head
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.block1 = ConvBlock(3, 32, pool=True)
        self.block2 = ConvBlock(32, 64, pool=True)
        self.block3 = ConvBlock(64, 128, pool=True)
        self.block4 = ConvBlock(128, 256, pool=True)
        self.block5 = ConvBlock(256, 256, pool=False)  # no pool, adds depth

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test
    for cls in [2, 5, 10]:
        small = MarioCNNSmall(num_classes=cls, img_size=64)
        med   = MarioCNNMedium(num_classes=cls)
        x64   = torch.randn(4, 3, 64, 64)
        x224  = torch.randn(4, 3, 224, 224)
        print(f"[{cls} classes] Small: {count_parameters(small):,} params | "
              f"out={small(x64).shape}")
        print(f"[{cls} classes] Medium: {count_parameters(med):,} params | "
              f"out={med(x224).shape}")
