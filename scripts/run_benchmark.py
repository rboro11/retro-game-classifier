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

FRAMES_DIR = ROOT / "data" / "processed" / "frames"
CKPT_DIR   = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"


def load_model_from_checkpoint(ckpt_path: Path):
    """Load a model from a checkpoint file using the saved config."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    model_name  = cfg["model_name"]
    num_classes = cfg["num_classes"]

    # Reconstruct model
    if model_name == "cnn_small":
        model = MarioCNNSmall(num_classes=num_classes)
    elif model_name == "cnn_medium":
        model = MarioCNNMedium(num_classes=num_classes)
    elif model_name == "audio_cnn":
        from models.audio_model import SpectrogramCNN
        model = SpectrogramCNN(num_classes=num_classes)
    elif model_name in MODEL_REGISTRY:
        model, _ = build_model(model_name, num_classes=num_classes)
    else:
        print(f"Warning: unknown model '{model_name}', skipping.")
        return None, None

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_name


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load class names from splits if available
    classes_csv = ROOT / "data" / "processed" / "splits" / "classes.csv"
    if classes_csv.exists():
        class_names = pd.read_csv(classes_csv)["label"].tolist()
    else:
        # Fallback: scan frames dir
        class_names = sorted([d.name for d in FRAMES_DIR.iterdir()
                              if d.is_dir()]) if FRAMES_DIR.exists() else None

    print(f"Classes: {class_names}")

    # Build test dataset
    test_ds = MarioImageDataset(FRAMES_DIR, split="test")
    test_loader = get_dataloader(test_ds, batch_size=32, num_workers=0)

    # Initialize benchmarker (auto-loads all *_history.json)
    bench = Benchmarker(str(CKPT_DIR), str(REPORTS_DIR))

    # Evaluate each saved model
    for ckpt_path in sorted(CKPT_DIR.glob("*_best.pt")):
        model, model_name = load_model_from_checkpoint(ckpt_path)
        if model is not None:
            bench.evaluate_model(model, test_loader, model_name, device)

    # Generate full report
    bench.run_all(test_loader, class_names, device)

    # Also save quick val-acc comparison (good for Colab inline display)
    plot_val_accuracy_comparison(str(CKPT_DIR),
                                 str(REPORTS_DIR / "val_acc_comparison.png"))
    print(f"\nAll reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
