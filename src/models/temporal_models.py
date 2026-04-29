"""
Temporal / Video Models — Phase 5

Three approaches for short video clips (N frames):
  1. CNN+LSTM    — CNN extracts frame features, LSTM aggregates over time
  2. 3D-ResNet   — Spatiotemporal convolutions (r3d_18 from torchvision)
  3. FrameAvg    — Simplest baseline: average CNN predictions across N frames
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import R3D_18_Weights
from typing import Tuple, Dict


# ─────────────────────────────────────────────
# 1. CNN + LSTM
# ─────────────────────────────────────────────

class MarioCNNLSTM(nn.Module):
    """
    Per-frame CNN (ResNet-18 backbone) → LSTM over time → classification.

    Input:  (B, T, C, H, W)  — T = num_frames
    Output: (B, num_classes)

    Notes:
    - Uses the pretrained ResNet-18 body as a feature extractor
    - LSTM hidden state from the last time step is classified
    - Good baseline temporal model, much lighter than 3D-ResNet
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.4,
                 freeze_cnn: bool = False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC; keep feature extractor (output: 512-d)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out_dim = 512

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Merge batch and time dims for CNN forward
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)          # (B*T, 512, 1, 1)
        feats = feats.view(B, T, -1) # (B, T, 512)
        # LSTM over time
        out, _ = self.lstm(feats)    # (B, T, hidden_dim)
        # Take last time step
        last = out[:, -1, :]         # (B, hidden_dim)
        return self.classifier(last)


# ─────────────────────────────────────────────
# 2. 3D-ResNet (r3d_18)
# ─────────────────────────────────────────────

def build_3d_resnet(num_classes: int,
                    freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    R3D-18: 3D Residual Network for video classification.
    Input: (B, C, T, H, W)  — T = num_frames (must be ≥ 8)
    Recommended T=8 or T=16.

    Heavier than CNN+LSTM but captures local spatiotemporal patterns directly.
    """
    model = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    meta = {
        "name": "3D-ResNet-18",
        "input_format": "(B, C, T, H, W)",
        "recommended_frames": 8,
        "img_size": 112,
        "trainable_params": sum(p.numel() for p in model.parameters()
                               if p.requires_grad),
    }
    return model, meta


# ─────────────────────────────────────────────
# 3. Frame Averaging Wrapper (simplest temporal baseline)
# ─────────────────────────────────────────────

class FrameAverageWrapper(nn.Module):
    """
    Applies any image classifier to each frame independently,
    then averages the logits (soft voting).

    This is the simplest 'temporal' model and a strong baseline.
    Useful to establish whether temporal info helps at all.

    Input:  (B, T, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.model = image_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.model(x)  # (B*T, num_classes)
        logits = logits.view(B, T, -1)  # (B, T, num_classes)
        return logits.mean(dim=1)  # (B, num_classes) — averaged over frames


# ─────────────────────────────────────────────
# Input reshaping utilities
# ─────────────────────────────────────────────

def frames_to_3d_input(frames: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, T, C, H, W) → (B, C, T, H, W) for 3D-ResNet.
    """
    return frames.permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)

    # CNN+LSTM
    cnn_lstm = MarioCNNLSTM(num_classes=5)
    print(f"CNN+LSTM output: {cnn_lstm(x).shape}")

    # 3D-ResNet
    r3d, meta = build_3d_resnet(num_classes=5)
    x_3d = frames_to_3d_input(x)  # (B, C, T, H, W)
    print(f"3D-ResNet output: {r3d(x_3d).shape}")

    # Frame average
    from transfer_models import build_model
    img_model, _ = build_model("resnet18", num_classes=5)
    avg_wrapper = FrameAverageWrapper(img_model)
    print(f"FrameAverage output: {avg_wrapper(x).shape}")
