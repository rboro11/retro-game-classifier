"""
Universal Trainer — works for any image/audio/video model.

Features:
  - Mixed-precision (AMP) support for Colab T4/V100
  - LR scheduler (CosineAnnealingLR or ReduceLROnPlateau)
  - Early stopping with patience
  - Checkpoint saving (best val accuracy)
  - Training history saved to JSON for the benchmarker
  - Label smoothing for better generalization
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# Training config dataclass
# ─────────────────────────────────────────────

class TrainConfig:
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 3,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",    # 'cosine' | 'plateau' | 'none'
        label_smoothing: float = 0.1,
        patience: int = 7,            # early stopping patience
        amp: bool = True,             # mixed precision
        save_dir: str = "checkpoints",
        device: Optional[str] = None,
    ):
        self.model_name    = model_name
        self.num_classes   = num_classes
        self.epochs        = epochs
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.scheduler     = scheduler
        self.label_smoothing = label_smoothing
        self.patience      = patience
        self.amp           = amp and torch.cuda.is_available()
        self.save_dir      = save_dir
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# ─────────────────────────────────────────────
# Trainer class
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model  = model.to(config.device)
        self.cfg    = config
        self.device = config.device

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )
        elif config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=3, factor=0.5
            )
        else:
            self.scheduler = None

        self.scaler = GradScaler(enabled=config.amp)

        # History
        self.history: Dict[str, List] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 0
        self.patience_counter = 0

        os.makedirs(config.save_dir, exist_ok=True)

    # ── one epoch ──────────────────────────────

    def _run_epoch(self, loader: DataLoader,
                   training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        correct    = 0
        total      = 0

        desc = "Train" if training else "Val"
        for batch in tqdm(loader, desc=desc, leave=False):
            # Support both image datasets (img, label)
            # and video datasets (frames, label)
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

            with torch.set_grad_enabled(training):
                with autocast(enabled=self.cfg.amp):
                    outputs = self.model(inputs)
                    loss    = self.criterion(outputs, labels)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item() * inputs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)

        return total_loss / total, correct / total

    # ── main training loop ──────────────────────

    def fit(self, train_loader: DataLoader,
            val_loader: DataLoader) -> Dict[str, List]:
        """
        Run the full training loop.
        Returns the history dict (also saved to save_dir/history.json).
        """
        print(f"\n{'='*60}")
        print(f"  Model:  {self.cfg.model_name}")
        print(f"  Device: {self.device}  |  AMP: {self.cfg.amp}")
        print(f"  Classes: {self.cfg.num_classes}  |  Epochs: {self.cfg.epochs}")
        print(f"{'='*60}\n")

        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   training=False)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            elif self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            print(f"Epoch [{epoch:3d}/{self.cfg.epochs}]  "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f}  |  "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.4f}  |  "
                  f"LR={current_lr:.2e}  [{elapsed:.1f}s]")

            # Checkpoint best model
            if val_acc > self.best_val_acc:
                self.best_val_acc   = val_acc
                self.best_epoch     = epoch
                self.patience_counter = 0
                ckpt_path = Path(self.cfg.save_dir) / f"{self.cfg.model_name}_best.pt"
                torch.save({
                    "epoch":      epoch,
                    "model_state": self.model.state_dict(),
                    "val_acc":    val_acc,
                    "config":     self.cfg.to_dict(),
                }, ckpt_path)
                print(f"  ✓ New best saved → {ckpt_path}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val_acc={self.best_val_acc:.4f} @ epoch {self.best_epoch})")
                break

        total_time = time.time() - t_start
        print(f"\nTraining complete in {total_time/60:.1f}min  "
              f"| Best val acc: {self.best_val_acc:.4f}")

        # Save history
        hist_path = Path(self.cfg.save_dir) / f"{self.cfg.model_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump({
                "config":  self.cfg.to_dict(),
                "history": self.history,
                "best_val_acc": self.best_val_acc,
                "best_epoch":   self.best_epoch,
                "total_time_min": round(total_time / 60, 2),
            }, f, indent=2)
        print(f"History saved → {hist_path}")

        return self.history

    def load_best(self) -> nn.Module:
        """Load the best checkpoint back into the model."""
        ckpt_path = Path(self.cfg.save_dir) / f"{self.cfg.model_name}_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
        return self.model
