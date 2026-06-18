"""
run_benchmark.py — generate comparison report across all trained models

Usage:
    python scripts/run_benchmark.py

Scans checkpoints/ for *_best.pt and *_history.json, loads each model,
runs inference on the test set, and generates:
  - reports/benchmark_report.png   (curves + confusion matrices + summary)
  - reports/benchmark_summary.csv
  - reports/per_class_report.csv
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset      import MarioImageDataset, get_dataloader
from models.cnn_custom import MarioCNNSmall, MarioCNNMedium
from models.transfer_models import build_model, MODEL_REGISTRY
from models.audio_model     import SpectrogramCNN, SpectrogramTransferNet
from evaluation.benchmarker import Benchmarker, plot_val_accuracy_comparison
import pandas as pd

FRAMES_DIR  = ROOT / "data" / "processed" / "frames"
CKPT_DIR    = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# Display-name → MODEL_REGISTRY key normalisation map.
#
# train_model.py saves meta["name"] (e.g. "ResNet-18", "EfficientNet-B0") into
# the checkpoint config, but MODEL_REGISTRY keys are lowercase/underscored
# (e.g. "resnet18", "efficientnet_b0").  This map handles both old checkpoints
# that used the display name and future ones that may already use the key.
#
# NOTE: slugification replaces ALL hyphens with underscores, so "ResNet-18"
# becomes "resnet_18" (not "resnet-18"). Both forms are listed below.
# ─────────────────────────────────────────────────────────────────────────────
_DISPLAY_TO_REGISTRY: dict[str, str] = {
    # Transfer models — display name produced by transfer_models.py meta dicts
    "resnet_18":          "resnet18",   # slug form of "ResNet-18"
    "resnet-18":          "resnet18",
    "resnet18":           "resnet18",
    "resnet_50":          "resnet50",   # slug form of "ResNet-50"
    "resnet-50":          "resnet50",
    "resnet50":           "resnet50",
    "efficientnet_b0":    "efficientnet_b0",
    "efficientnet-b0":    "efficientnet_b0",
    "efficientnet_b2":    "efficientnet_b2",
    "efficientnet-b2":    "efficientnet_b2",
    "efficientnet_b3":    "efficientnet_b3",
    "efficientnet-b3":    "efficientnet_b3",
    "mobilenetv3_small":  "mobilenet_v3",  # slug form of "MobileNetV3-Small"
    "mobilenetv3-small":  "mobilenet_v3",
    "mobilenet_v3":       "mobilenet_v3",
    "vit_b_16":           "vit_b16",    # slug form of "ViT-B/16"
    "vit-b/16":           "vit_b16",
    "vit_b16":            "vit_b16",
    # Custom / audio — already use registry keys, included for completeness
    "cnn_small":          "cnn_small",
    "cnn_medium":         "cnn_medium",
    "audio_cnn":          "audio_cnn",
    "audio_transfer":     "audio_transfer",
}


def _normalise_model_name(raw_name: str) -> str:
    """
    Convert a checkpoint model_name (display or registry form) to the
    canonical registry key used for model construction.

    Normalisation steps
    -------------------
    1. Strip surrounding whitespace.
    2. Lower-case.
    3. Replace spaces and hyphens with underscores for a consistent slug.
    4. Look up in the explicit _DISPLAY_TO_REGISTRY map first (handles
       non-obvious aliases like "ResNet-18" → "resnet18").
    5. If not found in the map, fall back to the slug itself — this future-
       proofs against new models added directly with registry-key names.
    """
    slug = raw_name.strip().lower().replace(" ", "_").replace("-", "_")

    if slug in _DISPLAY_TO_REGISTRY:
        return _DISPLAY_TO_REGISTRY[slug]

    return slug


def load_model_from_checkpoint(ckpt_path: Path):
    """Load a model from a checkpoint file using the saved config."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    raw_name    = cfg["model_name"]
    num_classes = cfg["num_classes"]

    model_key = _normalise_model_name(raw_name)

    # ── Custom CNN models ──────────────────────────────────────────
    if model_key == "cnn_small":
        model = MarioCNNSmall(num_classes=num_classes)

    elif model_key == "cnn_medium":
        model = MarioCNNMedium(num_classes=num_classes)

    # ── Audio models ─────────────────────────────────────────────
    elif model_key == "audio_cnn":
        model = SpectrogramCNN(num_classes=num_classes)

    elif model_key == "audio_transfer":
        model = SpectrogramTransferNet(num_classes=num_classes)

    # ── Transfer / pretrained models (all registered in MODEL_REGISTRY) ──────
    elif model_key in MODEL_REGISTRY:
        model, _ = build_model(model_key, num_classes=num_classes)

    else:
        print(
            f"Warning: cannot load '{raw_name}' "
            f"(normalised to '{model_key}') — not in MODEL_REGISTRY. Skipping.\n"
            f"  Known registry keys: {sorted(MODEL_REGISTRY.keys())}"
        )
        return None, None

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, raw_name


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load class names from splits if available
    classes_csv = ROOT / "data" / "processed" / "splits" / "classes.csv"
    if classes_csv.exists():
        class_names = pd.read_csv(classes_csv)["label"].tolist()
    else:
        class_names = sorted([d.name for d in FRAMES_DIR.iterdir()
                              if d.is_dir()]) if FRAMES_DIR.exists() else None

    print(f"Classes: {class_names}")

    # Build test dataset
    test_ds     = MarioImageDataset(FRAMES_DIR, split="test")
    test_loader = get_dataloader(test_ds, batch_size=32, num_workers=0)

    # Initialise benchmarker (auto-loads all *_history.json)
    bench = Benchmarker(str(CKPT_DIR), str(REPORTS_DIR))

    # Evaluate each saved checkpoint
    loaded = 0
    skipped = 0
    for ckpt_path in sorted(CKPT_DIR.glob("*_best.pt")):
        model, model_name = load_model_from_checkpoint(ckpt_path)
        if model is not None:
            bench.evaluate_model(model, test_loader, model_name, device)
            loaded += 1
        else:
            skipped += 1

    print(f"\nCheckpoints loaded: {loaded}  |  skipped: {skipped}")

    if loaded == 0:
        print("No models loaded — cannot generate report.")
        return

    # Generate full report
    bench.run_all(test_loader, class_names, device)

    # Quick val-acc comparison chart
    plot_val_accuracy_comparison(
        str(CKPT_DIR),
        str(REPORTS_DIR / "val_acc_comparison.png"),
    )
    print(f"\nAll reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
