"""
Transfer Learning Models — Phase 2/3 (image classification)

All models:
  - Accept num_classes to swap the classification head
  - Support two freeze modes: 'frozen' (head only) and 'full' (full fine-tune)
  - Return the model + a dict of metadata (params, expected input size, etc.)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    EfficientNet_B0_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    MobileNet_V3_Small_Weights,
)
from typing import Tuple, Dict


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final classification layer."""
    for name, param in model.named_parameters():
        if "classifier" in name or "fc" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# ResNet-18  (lightest pretrained option)
# ─────────────────────────────────────────────

def build_resnet18(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    ResNet-18 pretrained on ImageNet, head replaced.
    Input: (B, 3, 224, 224).
    Recommended for Phase 2 (fast training, decent accuracy).
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        _freeze_backbone(model)
    meta = {
        "name": "ResNet-18",
        "img_size": 224,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# ResNet-50  (Phase 3 workhorse)
# ─────────────────────────────────────────────

def build_resnet50(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    ResNet-50 pretrained on ImageNet, head replaced.
    Input: (B, 3, 224, 224).
    Best overall accuracy/compute trade-off for multi-era Mario.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        _freeze_backbone(model)
    meta = {
        "name": "ResNet-50",
        "img_size": 224,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# EfficientNet-B0  (best accuracy/size ratio)
# ─────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    EfficientNet-B0, fastest in the Efficient family.
    Input: (B, 3, 224, 224).
    Reference paper found this architecture family excels at game screenshot ID.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        _freeze_backbone(model)
    meta = {
        "name": "EfficientNet-B0",
        "img_size": 224,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# EfficientNet-B3  (top performer per arXiv 2311.15963)
# ─────────────────────────────────────────────

def build_efficientnet_b3(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    EfficientNet-B3: best single-model accuracy for game screenshot classification
    per arXiv 2311.15963 (77.67% over 8,796 games).
    Input: (B, 3, 300, 300) — use img_size=300 in transforms.
    """
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        _freeze_backbone(model)
    meta = {
        "name": "EfficientNet-B3",
        "img_size": 300,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# MobileNetV3-Small  (lightweight / phone inference)
# ─────────────────────────────────────────────

def build_mobilenet_v3(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    MobileNetV3-Small: extremely fast, good for CPU/mobile deployment.
    Input: (B, 3, 224, 224).
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    if freeze:
        _freeze_backbone(model)
    meta = {
        "name": "MobileNetV3-Small",
        "img_size": 224,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# Vision Transformer ViT-B/16  (Phase 3 / advanced)
# ─────────────────────────────────────────────

def build_vit_b16(num_classes: int, freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    ViT-B/16 pretrained on ImageNet-21k → ImageNet-1k.
    Best long-range texture / style reasoning. Heavier compute.
    Input: (B, 3, 224, 224).
    Requires: pip install timm
    """
    try:
        import timm
    except ImportError:
        raise ImportError("Install timm for ViT: pip install timm")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    if freeze:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    meta = {
        "name": "ViT-B/16",
        "img_size": 224,
        "trainable_params": count_trainable(model),
    }
    return model, meta


# ─────────────────────────────────────────────
# Model registry — single entry point
# ─────────────────────────────────────────────

MODEL_REGISTRY = {
    "resnet18":       build_resnet18,
    "resnet50":       build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b3": build_efficientnet_b3,
    "mobilenet_v3":   build_mobilenet_v3,
    "vit_b16":        build_vit_b16,
}


def build_model(name: str, num_classes: int,
                freeze: bool = False) -> Tuple[nn.Module, Dict]:
    """
    Build any registered model by name.

    Example:
        model, meta = build_model("resnet50", num_classes=5)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, freeze=freeze)


if __name__ == "__main__":
    for name in ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3"]:
        model, meta = build_model(name, num_classes=10)
        x = torch.randn(2, 3, meta["img_size"], meta["img_size"])
        out = model(x)
        print(f"{meta['name']:25s} | trainable: {meta['trainable_params']:>10,} | "
              f"output: {out.shape}")
