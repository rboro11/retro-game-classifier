"""
Late Fusion Multi-Modal Model — Phase 6

Combines image + audio predictions into a single final prediction.
Three fusion strategies:
  1. AverageFusion — simple weighted average of logits
  2. ConcatFusion  — concatenate embeddings → MLP head
  3. AttentionFusion — learned modality attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AverageFusion(nn.Module):
    """
    Weighted average of logits from N modalities.
    Weights are learnable scalars (initialized to equal weights).

    Usage:
        fusion = AverageFusion(num_modalities=2)
        logits = fusion([image_logits, audio_logits])
    """

    def __init__(self, num_modalities: int = 2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities))

    def forward(self, logits_list: list) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        out = sum(w[i] * logits_list[i] for i in range(len(logits_list)))
        return out


class ConcatFusion(nn.Module):
    """
    Concatenates penultimate-layer embeddings from each modality,
    then passes through a shared MLP classification head.

    Args:
        embed_dims: list of embedding dimensions per modality
                    e.g. [512, 512] for two ResNets
        num_classes: number of output classes
        hidden_dim:  MLP hidden size
        dropout:     dropout rate

    The backbone models are passed in at construction so gradients flow through.
    """

    def __init__(self, embed_dims: list, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        total_dim = sum(embed_dims)
        self.head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: list) -> torch.Tensor:
        concat = torch.cat(embeddings, dim=1)  # (B, sum(embed_dims))
        return self.head(concat)


class AttentionFusion(nn.Module):
    """
    Soft-attention fusion: learns which modality to trust more per sample.

    Each modality produces a logit vector; a small attention network
    computes per-modality weights conditioned on the logits themselves.

    Args:
        num_classes:    shared number of output classes
        num_modalities: number of input modality streams
    """

    def __init__(self, num_classes: int, num_modalities: int = 2):
        super().__init__()
        # Context vector for attention over modalities
        self.attn = nn.Sequential(
            nn.Linear(num_classes * num_modalities, num_modalities),
            nn.Softmax(dim=-1),
        )
        self.num_modalities = num_modalities

    def forward(self, logits_list: list) -> torch.Tensor:
        # logits_list: each (B, num_classes)
        stacked = torch.stack(logits_list, dim=1)  # (B, M, num_classes)
        B, M, C = stacked.shape
        flat = stacked.view(B, M * C)               # (B, M*C)
        weights = self.attn(flat)                   # (B, M)
        weights = weights.unsqueeze(-1)             # (B, M, 1)
        out = (stacked * weights).sum(dim=1)        # (B, num_classes)
        return out


# ─────────────────────────────────────────────
# Full multi-modal pipeline wrapper
# ─────────────────────────────────────────────

class MarioMultiModalClassifier(nn.Module):
    """
    End-to-end multi-modal classifier.
    Wraps separate image and audio backbones + a fusion module.

    Args:
        image_model:   any image classification model (outputs logits)
        audio_model:   any audio classification model (outputs logits)
        fusion:        one of 'average', 'concat', 'attention'
        num_classes:   number of game classes
        image_embed_dim: embedding dim from image backbone (for concat fusion)
        audio_embed_dim: embedding dim from audio backbone (for concat fusion)
    """

    def __init__(self,
                 image_model: nn.Module,
                 audio_model: nn.Module,
                 fusion: str = "average",
                 num_classes: int = 10,
                 image_embed_dim: int = 512,
                 audio_embed_dim: int = 512):
        super().__init__()
        self.image_model = image_model
        self.audio_model = audio_model
        self.fusion_type = fusion

        if fusion == "average":
            self.fusion_module = AverageFusion(num_modalities=2)
        elif fusion == "concat":
            self.fusion_module = ConcatFusion(
                embed_dims=[image_embed_dim, audio_embed_dim],
                num_classes=num_classes,
            )
        elif fusion == "attention":
            self.fusion_module = AttentionFusion(
                num_classes=num_classes, num_modalities=2
            )
        else:
            raise ValueError(f"fusion must be 'average', 'concat', or 'attention'")

    def forward(self, image: torch.Tensor,
                audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W)
            audio: (B, 3, H, W) mel-spectrogram, or None (image-only inference)
        """
        img_logits = self.image_model(image)

        if audio is None:
            return img_logits  # graceful fallback

        aud_logits = self.audio_model(audio)

        if self.fusion_type in ("average", "attention"):
            return self.fusion_module([img_logits, aud_logits])
        else:
            # concat fusion needs embeddings, not logits
            # in this case models should return embeddings — override as needed
            return self.fusion_module([img_logits, aud_logits])


if __name__ == "__main__":
    NUM_CLASSES = 5
    B = 4

    dummy_img_model = nn.Linear(10, NUM_CLASSES)
    dummy_aud_model = nn.Linear(10, NUM_CLASSES)

    for fusion_type in ["average", "concat", "attention"]:
        img_logits = torch.randn(B, NUM_CLASSES)
        aud_logits = torch.randn(B, NUM_CLASSES)

        if fusion_type == "average":
            f = AverageFusion(2)
            out = f([img_logits, aud_logits])
        elif fusion_type == "concat":
            f = ConcatFusion([NUM_CLASSES, NUM_CLASSES], NUM_CLASSES)
            out = f([img_logits, aud_logits])
        elif fusion_type == "attention":
            f = AttentionFusion(NUM_CLASSES, 2)
            out = f([img_logits, aud_logits])

        print(f"Fusion={fusion_type}: output shape = {out.shape}")
