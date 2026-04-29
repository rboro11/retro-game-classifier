"""
Audio Classifier — Phase 4

Treats mel-spectrograms as images and uses a CNN backbone.
Two options:
  1. SpectrogramCNN — lightweight custom CNN for audio
  2. SpectrogramTransferNet — EfficientNet-B0 fine-tuned on spectrograms

Input to both: mel-spectrogram tensor (B, 3, 128, time_frames)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class SpectrogramCNN(nn.Module):
    """
    Small CNN designed for mel-spectrogram inputs.
    3 convolutional blocks with increasing receptive field.

    Input:  (B, 3, 128, T)   — 3-channel replicated mel-spectrogram
    Output: (B, num_classes)

    Lightweight; trains fast even on CPU.
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: capture low-level spectral patterns
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),   # halve freq and time dims

            # Block 2: mid-level temporal patterns
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 3: higher-level song structure
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SpectrogramTransferNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned on mel-spectrograms.
    Accepts (B, 3, H, W) spectrogram images (H=128, W=variable).
    More powerful than SpectrogramCNN but requires more data.
    """

    def __init__(self, num_classes: int, freeze: bool = False,
                 dropout: float = 0.4):
        super().__init__()
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = base

        if freeze:
            for name, p in self.model.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Simulate a batch of spectrograms: (B, 3, 128, 216) — ~5 sec clip
    x = torch.randn(4, 3, 128, 216)

    for cls in [3, 10]:
        small = SpectrogramCNN(num_classes=cls)
        big   = SpectrogramTransferNet(num_classes=cls)
        print(f"[{cls} classes] SpectrogramCNN: {small(x).shape}")
        print(f"[{cls} classes] SpectrogramTransfer: {big(x).shape}")
