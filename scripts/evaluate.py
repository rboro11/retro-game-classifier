"""
evaluate.py — honest held-out test set evaluation for all saved checkpoints

Usage:
    python scripts/evaluate.py                        # evaluate all *_best.pt
    python scripts/evaluate.py --model cnn_small      # evaluate one model
    python scripts/evaluate.py --model ResNet-18      # transfer model by name

Outputs per model:
  - Full classification report (precision / recall / F1 / support)
  - Confusion matrix saved to reports/eval_<model_name>_confusion.png
  - Per-model row appended to reports/eval_summary.csv

The test split is loaded from data/processed/splits/test.csv so the same
held-out frames are used every time, regardless of how the dataset is rebuilt.
"""

import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import MarioImageDataset, get_dataloader
from models.cnn_custom import MarioCNNSmall, MarioCNNMedium
from models.transfer_models import build_model, MODEL_REGISTRY
from models.audio_model import SpectrogramCNN, SpectrogramTransferNet

FRAMES_DIR  = ROOT / "data" / "processed" / "frames"
CKPT_DIR    = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────────
DISPLAY_TO_REGISTRY = {
    "resnet_18": "resnet18", "resnet-18": "resnet18", "resnet18": "resnet18",
    "resnet_50": "resnet50", "resnet-50": "resnet50", "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0", "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2", "efficientnet-b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3", "efficientnet-b3": "efficientnet_b3",
    "mobilenetv3_small": "mobilenet_v3",  "mobilenet_v3": "mobilenet_v3",
    "vit_b_16": "vit_b16", "vit-b/16": "vit_b16", "vit_b16": "vit_b16",
    "cnn_small": "cnn_small", "cnn_medium": "cnn_medium",
    "audio_cnn": "audio_cnn", "audio_transfer": "audio_transfer",
}


def normalise(name: str) -> str:
    slug = name.strip().lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_TO_REGISTRY.get(slug, slug)


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    key  = normalise(cfg["model_name"])
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
        print(f"  ⚠ Unknown model key '{key}' — skipping {ckpt_path.name}")
        return None, None, None

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg["model_name"], cfg.get("img_size", 224)


def run_inference(model, loader, device):
    model = model.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[1].set_title("Normalised (row %%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    out_path  = out_dir / f"eval_{safe_name}_confusion.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix → {out_path}")
    return out_path


def evaluate_one(ckpt_path: Path, device: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  Checkpoint: {ckpt_path.name}")

    model, model_name, img_size = load_model(ckpt_path)
    if model is None:
        return None

    print(f"  Model: {model_name}  |  img_size: {img_size}")

    # Build test dataset from the frozen test.csv split
    test_ds = MarioImageDataset(FRAMES_DIR, split="test", img_size=img_size)
    test_loader = get_dataloader(test_ds, batch_size=64, num_workers=0)
    class_names = test_ds.classes

    print(f"  Test samples: {len(test_ds):,}  |  Classes: {class_names}")

    y_true, y_pred = run_inference(model, test_loader, device)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                zero_division=0))

    save_confusion_matrix(y_true, y_pred, class_names, model_name, REPORTS_DIR)

    return {
        "model":     model_name,
        "test_acc":  round(acc,  4),
        "macro_f1":  round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "n_test":    len(y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Evaluate one model by name (e.g. cnn_small, ResNet-18). "
                             "Omit to evaluate all *_best.pt checkpoints.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.model:
        # Find the checkpoint matching the requested model name
        safe  = args.model.replace("/", "_").replace(" ", "_")
        candidates = list(CKPT_DIR.glob(f"*{safe}*_best.pt")) + \
                     list(CKPT_DIR.glob(f"*{args.model}*_best.pt"))
        candidates = list({p.resolve() for p in candidates})  # deduplicate
        if not candidates:
            print(f"No checkpoint found matching '{args.model}' in {CKPT_DIR}")
            print(f"Available: {[p.name for p in sorted(CKPT_DIR.glob('*_best.pt'))]}")
            return
        ckpt_paths = candidates
    else:
        ckpt_paths = sorted(CKPT_DIR.glob("*_best.pt"))
        if not ckpt_paths:
            print(f"No *_best.pt files found in {CKPT_DIR}")
            return

    results = []
    for ckpt_path in ckpt_paths:
        row = evaluate_one(ckpt_path, device)
        if row:
            results.append(row)

    if not results:
        print("\nNo models evaluated successfully.")
        return

    # Save summary CSV
    summary_df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
    csv_path   = REPORTS_DIR / "eval_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved → {csv_path}")


if __name__ == "__main__":
    main()
