"""
export_model.py — package a trained checkpoint into a self-contained inference bundle

Produces a single .pt file that contains everything the inference app needs:
  - model architecture key and num_classes
  - model weights (state_dict)
  - class names in label-index order
  - transform config (img_size, mean, std)
  - training metadata (val_acc, test_acc, epoch, train_time)

The exported bundle can be loaded with a single torch.load() call — no
access to training code required.

Usage:
    # Export the best model by test accuracy (default)
    python scripts/export_model.py

    # Export a specific model
    python scripts/export_model.py --model EfficientNet-B0

    # Export with a specific test accuracy to embed in the bundle
    python scripts/export_model.py --model EfficientNet-B0 --test_acc 1.0

    # Choose output path
    python scripts/export_model.py --output exports/smb_classifier.pt

Output:
    exports/<model_name>_export.pt     (self-contained inference bundle)
    exports/<model_name>_export.json   (human-readable metadata sidecar)
"""

import sys
import json
import argparse
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import MarioImageDataset
from models.cnn_custom import MarioCNNSmall, MarioCNNMedium
from models.transfer_models import build_model, MODEL_REGISTRY
from models.audio_model import SpectrogramCNN, SpectrogramTransferNet

CKPT_DIR    = ROOT / "checkpoints"
EXPORTS_DIR = ROOT / "exports"
REPORTS_DIR = ROOT / "reports"
FRAMES_DIR  = ROOT / "data" / "processed" / "frames"

EXPORTS_DIR.mkdir(exist_ok=True)

# ImageNet normalisation — matches training transforms in dataset.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DISPLAY_TO_REGISTRY = {
    "resnet_18": "resnet18", "resnet-18": "resnet18", "resnet18": "resnet18",
    "resnet_50": "resnet50", "resnet-50": "resnet50", "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0", "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2", "efficientnet-b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3", "efficientnet-b3": "efficientnet_b3",
    "mobilenetv3_small": "mobilenet_v3", "mobilenet_v3": "mobilenet_v3",
    "vit_b_16": "vit_b16", "vit-b/16": "vit_b16", "vit_b16": "vit_b16",
    "cnn_small": "cnn_small", "cnn_medium": "cnn_medium",
    "audio_cnn": "audio_cnn", "audio_transfer": "audio_transfer",
}


def normalise_key(name: str) -> str:
    slug = name.strip().lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_TO_REGISTRY.get(slug, slug)


def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    key  = normalise_key(cfg["model_name"])
    n    = cfg["num_classes"]

    if key == "cnn_small":
        model = MarioCNNSmall(num_classes=n)
    elif key == "cnn_medium":
        model = MarioCNNMedium(num_classes=n)
    elif key == "audio_cnn":
        model = SpectrogramCNN(num_classes=n)
    elif key == "audio_transfer":
        model = SpectrogramTransferNet(num_classes=n)
    elif key in MODEL_REGISTRY:
        model, _ = build_model(key, num_classes=n)
    else:
        raise ValueError(f"Unknown model key '{key}' in checkpoint config.")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, key, cfg


def pick_best_checkpoint():
    """
    Return the checkpoint with the highest test_acc from eval_summary.csv,
    or fall back to highest val_acc from history JSONs.
    """
    eval_csv = REPORTS_DIR / "eval_summary.csv"
    if eval_csv.exists():
        import pandas as pd
        df = pd.read_csv(eval_csv).sort_values("test_acc", ascending=False)
        best_model_name = df.iloc[0]["model"]
        print(f"Best model by test accuracy: {best_model_name} "
              f"({df.iloc[0]['test_acc']:.4f})")
        candidates = list(CKPT_DIR.glob(f"*{best_model_name}*_best.pt"))
        if candidates:
            return candidates[0], float(df.iloc[0]["test_acc"])

    # Fallback: pick by val_acc from history JSONs
    best_acc, best_path = 0.0, None
    for hist_file in CKPT_DIR.glob("*_history.json"):
        with open(hist_file) as f:
            data = json.load(f)
        acc = data.get("best_val_acc", 0)
        if acc > best_acc:
            best_acc = acc
            model_name = hist_file.stem.replace("_history", "")
            ckpt = CKPT_DIR / f"{model_name}_best.pt"
            if ckpt.exists():
                best_path = ckpt
    return best_path, best_acc


def get_class_names():
    """Read class names from the dataset directory structure."""
    try:
        ds = MarioImageDataset(FRAMES_DIR, split="test")
        return ds.classes
    except Exception:
        pass
    # Fallback: sorted subdirectory names
    subdirs = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
    return subdirs if subdirs else ["SMB1", "SMB3"]


def export(ckpt_path: Path, output_path: Path, test_acc: float = None):
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    model, ckpt, registry_key, cfg = load_checkpoint(ckpt_path)

    class_names = get_class_names()
    img_size    = cfg.get("img_size", 224)

    # Load history for metadata
    hist_file = CKPT_DIR / f"{cfg['model_name']}_history.json"
    train_time = None
    if hist_file.exists():
        with open(hist_file) as f:
            hist = json.load(f)
        train_time = hist.get("total_time_min")

    bundle = {
        # Weights
        "model_state": model.state_dict(),

        # Architecture — enough for any loader to rebuild the model
        "architecture": {
            "registry_key": registry_key,       # e.g. 'efficientnet_b0'
            "display_name": cfg["model_name"],  # e.g. 'EfficientNet-B0'
            "num_classes":  cfg["num_classes"],
        },

        # Inference config
        "inference": {
            "img_size":  img_size,
            "mean":      IMAGENET_MEAN,
            "std":       IMAGENET_STD,
            "class_names": class_names,
        },

        # Provenance
        "training": {
            "best_val_acc":    ckpt.get("val_acc"),
            "best_epoch":      ckpt.get("epoch"),
            "train_time_min":  train_time,
            "test_acc":        test_acc,
            "num_classes":     cfg["num_classes"],
            "original_ckpt":   ckpt_path.name,
        },
    }

    torch.save(bundle, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✓ Exported → {output_path}  ({size_mb:.1f} MB)")

    # Human-readable sidecar JSON (weights excluded)
    sidecar = {
        "architecture": bundle["architecture"],
        "inference":    bundle["inference"],
        "training":     bundle["training"],
        "file_size_mb": round(size_mb, 2),
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"✓ Metadata sidecar → {json_path}")

    print("\nBundle contents:")
    print(f"  architecture : {bundle['architecture']}")
    print(f"  class_names  : {class_names}")
    print(f"  img_size     : {img_size}")
    print(f"  val_acc      : {ckpt.get('val_acc', '?')}")
    print(f"  test_acc     : {test_acc if test_acc is not None else 'not set'}")
    print(f"  size         : {size_mb:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model name to export (e.g. EfficientNet-B0). "
                             "Omit to auto-select the best model by test accuracy.")
    parser.add_argument("--test_acc", type=float, default=None,
                        help="Test accuracy to embed in the bundle metadata.")
    parser.add_argument("--output", default=None,
                        help="Output .pt path. Defaults to exports/<model>_export.pt")
    args = parser.parse_args()

    if args.model:
        candidates = list(CKPT_DIR.glob(f"*{args.model}*_best.pt"))
        if not candidates:
            print(f"No checkpoint matching '{args.model}' in {CKPT_DIR}")
            print(f"Available: {[p.name for p in sorted(CKPT_DIR.glob('*_best.pt'))]}")
            return
        ckpt_path = candidates[0]
        test_acc  = args.test_acc
    else:
        ckpt_path, test_acc = pick_best_checkpoint()
        if args.test_acc is not None:
            test_acc = args.test_acc
        if ckpt_path is None:
            print(f"No checkpoints found in {CKPT_DIR}")
            return

    safe_name = ckpt_path.stem.replace("_best", "")
    out_path  = Path(args.output) if args.output else EXPORTS_DIR / f"{safe_name}_export.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export(ckpt_path, out_path, test_acc=test_acc)


if __name__ == "__main__":
    main()
