"""
Model Benchmarker — Phase comparison harness

Loads all saved training histories and test predictions,
then produces:
  1. Side-by-side accuracy/loss training curves
  2. Confusion matrices per model
  3. Per-class precision / recall / F1 table
  4. Summary comparison table (val acc, test acc, params, train time)
  5. Saved report: reports/benchmark_report.png + benchmark_summary.csv

Usage:
    from src.evaluation.benchmarker import Benchmarker
    bench = Benchmarker(checkpoints_dir="checkpoints")
    bench.run_all(test_loader, class_names)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import seaborn as sns


class Benchmarker:
    """
    Collects results from multiple trained models and generates
    a comprehensive visual + tabular comparison.
    """

    def __init__(self, checkpoints_dir: str = "checkpoints",
                 reports_dir: str = "reports"):
        self.ckpt_dir   = Path(checkpoints_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        self.histories  : Dict[str, dict] = {}   # model_name → history dict
        self.predictions: Dict[str, dict] = {}   # model_name → {y_true, y_pred}
        self._load_histories()

    # ── Load saved histories ─────────────────────

    def _load_histories(self):
        for hist_file in self.ckpt_dir.glob("*_history.json"):
            model_name = hist_file.stem.replace("_history", "")
            with open(hist_file) as f:
                data = json.load(f)
            self.histories[model_name] = data
            print(f"Loaded history: {model_name} "
                  f"(best_val_acc={data.get('best_val_acc', '?'):.4f})")

    # ── Evaluate a model on test set ─────────────

    def evaluate_model(self, model: torch.nn.Module,
                       test_loader, model_name: str,
                       device: str = "cpu"):
        """
        Run inference on test_loader and store predictions.
        Call this for each model before generating the report.
        """
        model = model.to(device)
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"{model_name}: test accuracy = {acc:.4f}")

        self.predictions[model_name] = {
            "y_true": all_labels,
            "y_pred": all_preds,
            "test_acc": acc,
        }

    # ── Training curves ──────────────────────────

    def plot_training_curves(self, ax_loss, ax_acc, model_name: str,
                             color: str):
        if model_name not in self.histories:
            return
        h = self.histories[model_name]["history"]
        epochs = range(1, len(h["train_loss"]) + 1)

        ax_loss.plot(epochs, h["train_loss"], "--", color=color, alpha=0.6,
                     label=f"{model_name} train")
        ax_loss.plot(epochs, h["val_loss"],   "-",  color=color, linewidth=2,
                     label=f"{model_name} val")

        ax_acc.plot(epochs, h["train_acc"], "--", color=color, alpha=0.6,
                    label=f"{model_name} train")
        ax_acc.plot(epochs, h["val_acc"],   "-",  color=color, linewidth=2,
                    label=f"{model_name} val")

    # ── Confusion matrix ─────────────────────────

    def plot_confusion_matrix(self, ax, model_name: str,
                              class_names: List[str]):
        if model_name not in self.predictions:
            ax.set_visible(False)
            return
        data = self.predictions[model_name]
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", ax=ax, cbar=False,
            annot_kws={"size": 8},
        )
        ax.set_title(f"{model_name}\nacc={data['test_acc']:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    # ── Summary table ────────────────────────────

    def build_summary_table(self, class_names: List[str]) -> pd.DataFrame:
        rows = []
        for model_name, hist_data in self.histories.items():
            cfg = hist_data.get("config", {})
            row = {
                "Model": model_name,
                "Best Val Acc": round(hist_data.get("best_val_acc", 0), 4),
                "Best Epoch":   hist_data.get("best_epoch", "?"),
                "Train Time (min)": hist_data.get("total_time_min", "?"),
            }
            if model_name in self.predictions:
                pred_data = self.predictions[model_name]
                row["Test Acc"] = round(pred_data["test_acc"], 4)
                report = classification_report(
                    pred_data["y_true"], pred_data["y_pred"],
                    target_names=class_names, output_dict=True, zero_division=0
                )
                row["Macro F1"] = round(report["macro avg"]["f1-score"], 4)
            else:
                row["Test Acc"] = "—"
                row["Macro F1"] = "—"
            rows.append(row)

        df = pd.DataFrame(rows)
        if "Test Acc" in df.columns:
            df = df.sort_values("Test Acc", ascending=False)
        return df

    # ── Per-class performance ────────────────────

    def per_class_report(self, class_names: List[str]) -> pd.DataFrame:
        """Returns a DataFrame with precision/recall/F1 per class per model."""
        dfs = []
        for model_name, pred_data in self.predictions.items():
            report = classification_report(
                pred_data["y_true"], pred_data["y_pred"],
                target_names=class_names, output_dict=True, zero_division=0
            )
            for cls_name in class_names:
                if cls_name in report:
                    r = report[cls_name]
                    dfs.append({
                        "Model":     model_name,
                        "Class":     cls_name,
                        "Precision": round(r["precision"], 4),
                        "Recall":    round(r["recall"],    4),
                        "F1":        round(r["f1-score"],  4),
                        "Support":   int(r["support"]),
                    })
        return pd.DataFrame(dfs)

    # ── Master report ────────────────────────────

    def run_all(self, test_loader=None,
                class_names: Optional[List[str]] = None,
                device: str = "cpu"):
        """
        Generate the full benchmark report.
        If models haven't been evaluated yet (via evaluate_model),
        the confusion matrix section will be empty.
        """
        model_names = list(self.histories.keys())
        if not model_names:
            print("No training histories found. Train at least one model first.")
            return

        if class_names is None:
            # Fallback: use integer labels
            num_classes = self.histories[model_names[0]]["config"].get("num_classes", 5)
            class_names = [str(i) for i in range(num_classes)]

        colors = plt.cm.tab10.colors
        n_models = len(model_names)

        # ── Layout ─────────────────────────────────
        # Row 0: training curves (loss | acc) spanning all models
        # Row 1: confusion matrices (one per model)
        # Row 2: summary table

        fig = plt.figure(figsize=(max(14, 5 * n_models), 18))
        fig.suptitle("Mario Game Identifier — Model Benchmark",
                     fontsize=16, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(3, max(n_models, 2), figure=fig,
                               hspace=0.5, wspace=0.35)

        # ── Training curves ─────────────────────────
        ax_loss = fig.add_subplot(gs[0, :n_models // 2 + 1])
        ax_acc  = fig.add_subplot(gs[0, n_models // 2 + 1:])

        for i, name in enumerate(model_names):
            self.plot_training_curves(ax_loss, ax_acc, name, colors[i % 10])

        ax_loss.set_title("Loss Curves")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross-Entropy Loss")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)

        ax_acc.set_title("Accuracy Curves")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(fontsize=8)
        ax_acc.grid(alpha=0.3)

        # ── Confusion matrices ───────────────────────
        for i, name in enumerate(model_names):
            ax_cm = fig.add_subplot(gs[1, i % max(n_models, 2)])
            self.plot_confusion_matrix(ax_cm, name, class_names)

        # ── Summary table (text in figure) ──────────
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis("off")
        summary_df = self.build_summary_table(class_names)
        col_labels = list(summary_df.columns)
        cell_text  = summary_df.values.tolist()
        tbl = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        ax_table.set_title("Summary Comparison", fontsize=12,
                            fontweight="bold", pad=10)

        # ── Save ────────────────────────────────────
        report_path = self.reports_dir / "benchmark_report.png"
        fig.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Benchmark report saved → {report_path}")

        # Save CSVs
        summary_df.to_csv(self.reports_dir / "benchmark_summary.csv", index=False)
        per_class_df = self.per_class_report(class_names)
        if not per_class_df.empty:
            per_class_df.to_csv(self.reports_dir / "per_class_report.csv", index=False)

        print(f"\n{'─'*60}")
        print(summary_df.to_string(index=False))
        print(f"{'─'*60}")

        return summary_df


# ─────────────────────────────────────────────
# Standalone plot: compare val accuracy curves
# ─────────────────────────────────────────────

def plot_val_accuracy_comparison(checkpoints_dir: str = "checkpoints",
                                 save_path: str = "reports/val_acc_comparison.png"):
    """
    Quick standalone plot: all models' validation accuracy on one chart.
    Great for Colab inline display.
    """
    ckpt_dir = Path(checkpoints_dir)
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    found = 0

    for i, hist_file in enumerate(sorted(ckpt_dir.glob("*_history.json"))):
        with open(hist_file) as f:
            data = json.load(f)
        model_name = hist_file.stem.replace("_history", "")
        h = data["history"]
        epochs = range(1, len(h["val_acc"]) + 1)
        best   = data.get("best_val_acc", max(h["val_acc"]))
        ax.plot(epochs, h["val_acc"], "-o", markersize=3,
                color=colors[i % 10], linewidth=2,
                label=f"{model_name} (best={best:.4f})")
        found += 1

    if found == 0:
        print("No history files found.")
        return

    ax.set_title("Validation Accuracy — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")
