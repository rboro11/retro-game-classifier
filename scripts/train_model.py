"""
train_model.py — single model training entry point

Usage examples:
    # Phase 1: binary custom CNN
    python scripts/train_model.py --model cnn_small --num_classes 2 --epochs 20

    # Phase 2: ResNet-18 fine-tune
    python scripts/train_model.py --model resnet18 --num_classes 3 --epochs 30 --lr 3e-4

    # Phase 3: EfficientNet-B3 (img_size auto-set to 300)
    python scripts/train_model.py --model efficientnet_b3 --num_classes 10 --epochs 40

    # Audio classifier
    python scripts/train_model.py --model audio_cnn --num_classes 5 --modality audio
"""

import sys
import argparse
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import (
    MarioImageDataset, MarioAudioDataset,
    get_image_transforms, get_nes_transforms, get_dataloader
)
from models.cnn_custom    import MarioCNNSmall, MarioCNNMedium
from models.transfer_models import build_model, MODEL_REGISTRY
from models.audio_model   import SpectrogramCNN, SpectrogramTransferNet
from training.trainer     import Trainer, TrainConfig


FRAMES_DIR = ROOT / "data" / "processed" / "frames"
AUDIO_DIR  = ROOT / "data" / "processed" / "audio"
CKPT_DIR   = ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="resnet18",
                   help=f"Model name. Choose: cnn_small, cnn_medium, audio_cnn, "
                        f"audio_transfer, or any of {list(MODEL_REGISTRY.keys())}")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--modality",    default="image",
                   choices=["image", "audio"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--freeze",      action="store_true",
                   help="Freeze backbone, train head only (faster for small datasets)")
    p.add_argument("--balanced",    action="store_true",
                   help="Use weighted sampler for class imbalance")
    p.add_argument("--img_size",    type=int, default=None,
                   help="Override image size (auto-set per model if not given)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Build model ─────────────────────────────
    if args.model == "cnn_small":
        img_size = args.img_size or 64
        model = MarioCNNSmall(num_classes=args.num_classes, img_size=img_size)
        meta  = {"name": "cnn_small", "img_size": img_size}

    elif args.model == "cnn_medium":
        img_size = args.img_size or 224
        model = MarioCNNMedium(num_classes=args.num_classes)
        meta  = {"name": "cnn_medium", "img_size": img_size}

    elif args.model == "audio_cnn":
        model = SpectrogramCNN(num_classes=args.num_classes)
        meta  = {"name": "audio_cnn", "img_size": 128}

    elif args.model == "audio_transfer":
        model = SpectrogramTransferNet(num_classes=args.num_classes,
                                        freeze=args.freeze)
        meta  = {"name": "audio_transfer", "img_size": 224}

    else:
        model, meta = build_model(args.model, args.num_classes,
                                  freeze=args.freeze)

    img_size = args.img_size or meta.get("img_size", 224)
    print(f"\nModel: {meta['name']} | img_size={img_size} | "
          f"classes={args.num_classes}")

    # ── Build datasets ───────────────────────────
    if args.modality == "image":
        data_root = FRAMES_DIR
        train_ds  = MarioImageDataset(data_root, split="train",
                                       img_size=img_size)
        val_ds    = MarioImageDataset(data_root, split="val",
                                       transform=get_image_transforms("val", img_size))
    else:
        data_root = AUDIO_DIR
        train_ds  = MarioAudioDataset(data_root, split="train")
        val_ds    = MarioAudioDataset(data_root, split="val")

    print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")
    print(f"Classes: {train_ds.classes}")

    train_loader = get_dataloader(train_ds, args.batch_size,
                                  balanced=args.balanced)
    val_loader   = get_dataloader(val_ds, args.batch_size)

    # ── Train ────────────────────────────────────
    cfg = TrainConfig(
        model_name=meta["name"],
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=str(CKPT_DIR),
    )
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
