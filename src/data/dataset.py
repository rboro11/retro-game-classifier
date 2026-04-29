"""
MarioDataset — plug-and-play dataset for all modalities.

Directory layout expected:
  data/raw/<ClassName>/         <- drop raw PNGs, JPGs, MP4s, WAVs here
  data/processed/frames/<ClassName>/   <- auto-populated by extract_frames.py
  data/processed/spectrograms/<ClassName>/ <- auto-populated by gen_spectrograms.py
  data/processed/splits/train.csv / val.csv / test.csv  <- auto by build_dataset.py

CSV format:  filepath,label,label_idx,modality
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchaudio


# ─────────────────────────────────────────────
# Default transforms
# ─────────────────────────────────────────────

def get_image_transforms(mode: str = "train", img_size: int = 224):
    """Standard ImageNet-style transforms. Works for screenshots & spectrograms."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_nes_transforms(mode: str = "train", img_size: int = 64):
    """Lighter-weight transforms for low-res NES frames (256×240)."""
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])


# ─────────────────────────────────────────────
# Image Dataset
# ─────────────────────────────────────────────

class MarioImageDataset(Dataset):
    """
    Loads screenshots/frames for image classification.
    Supports both CSV-based loading (preferred) and folder-scan fallback.
    """

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, img_size: int = 224):
        """
        Args:
            root_dir:  path to data/processed/frames/ OR data/raw/
            split:     'train' | 'val' | 'test'
            transform: torchvision transforms (auto-selected if None)
            img_size:  resize target (default 224 for pretrained models)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or get_image_transforms(split, img_size)

        # Try CSV split first, fall back to folder scan
        csv_path = self.root_dir.parent.parent / "processed" / "splits" / f"{split}.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["modality"] == "image"].reset_index(drop=True)
        else:
            self.df = self._scan_folders()

        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.df["label_idx"] = self.df["label"].map(self.class_to_idx)

    def _scan_folders(self) -> pd.DataFrame:
        """Fallback: scan root_dir/<ClassName>/ for image files."""
        records = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for fpath in class_dir.glob("*"):
                if fpath.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    records.append({
                        "filepath": str(fpath),
                        "label": class_dir.name,
                        "modality": "image",
                    })
        return pd.DataFrame(records)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label_idx"])
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Returns per-sample weights for WeightedRandomSampler (handles class imbalance)."""
        counts = self.df["label_idx"].value_counts().sort_index()
        weights = 1.0 / counts
        sample_weights = self.df["label_idx"].map(weights).values
        return torch.FloatTensor(sample_weights)


# ─────────────────────────────────────────────
# Audio Dataset (mel-spectrogram)
# ─────────────────────────────────────────────

class MarioAudioDataset(Dataset):
    """
    Loads .wav audio clips and converts to mel-spectrograms on the fly.
    Expects: data/processed/spectrograms/<ClassName>/<file>.wav
    OR pre-computed PNGs in the same structure.
    """

    MEL_CONFIG = dict(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000,
    )

    def __init__(self, root_dir: str, split: str = "train",
                 clip_duration: float = 5.0, transform=None):
        """
        Args:
            root_dir:      path to processed audio directory
            split:         'train' | 'val' | 'test'
            clip_duration: seconds to sample from each file
            transform:     applied to the spectrogram image (Tensor)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_samples = int(clip_duration * self.MEL_CONFIG["sample_rate"])
        self.transform = transform

        # Build mel-spectrogram transform (torchaudio)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.MEL_CONFIG["sample_rate"],
            n_fft=self.MEL_CONFIG["n_fft"],
            hop_length=self.MEL_CONFIG["hop_length"],
            n_mels=self.MEL_CONFIG["n_mels"],
            f_min=self.MEL_CONFIG["f_min"],
            f_max=self.MEL_CONFIG["f_max"],
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.df = self._scan_folders()
        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.df["label_idx"] = self.df["label"].map(self.class_to_idx)

    def _scan_folders(self):
        records = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for fpath in class_dir.glob("*.wav"):
                records.append({"filepath": str(fpath), "label": class_dir.name})
        return pd.DataFrame(records)

    def _load_clip(self, filepath: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(filepath)
        # Resample if needed
        if sr != self.MEL_CONFIG["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, self.MEL_CONFIG["sample_rate"])
            waveform = resampler(waveform)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Pad or crop to fixed length
        n = waveform.shape[1]
        if n < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - n))
        else:
            start = torch.randint(0, n - self.clip_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.clip_samples]
        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._load_clip(row["filepath"])
        mel = self.mel_transform(waveform)        # (1, n_mels, time)
        mel = self.amplitude_to_db(mel)
        # Normalize to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        # Repeat to 3 channels so pretrained CNNs work
        mel = mel.repeat(3, 1, 1)
        if self.transform:
            mel = self.transform(mel)
        label = int(row["label_idx"])
        return mel, label


# ─────────────────────────────────────────────
# Video Dataset (frame stack)
# ─────────────────────────────────────────────

class MarioVideoDataset(Dataset):
    """
    Loads short video clips as a stack of N evenly-sampled frames.
    Expects: data/raw/<ClassName>/<clip>.mp4
    """

    def __init__(self, root_dir: str, split: str = "train",
                 num_frames: int = 8, img_size: int = 112,
                 transform=None):
        """
        Args:
            root_dir:   path to raw video directory
            split:      'train' | 'val' | 'test'
            num_frames: frames to sample per clip
            img_size:   spatial resize target
        """
        try:
            import torchvision.io as tvio
            self._tvio = tvio
        except ImportError:
            raise ImportError("torchvision with video support required: pip install torchvision")

        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform or get_image_transforms(split, img_size)

        self.df = self._scan_folders()
        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.df["label_idx"] = self.df["label"].map(self.class_to_idx)

    def _scan_folders(self):
        records = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for fpath in class_dir.glob("*.mp4"):
                records.append({"filepath": str(fpath), "label": class_dir.name})
        return pd.DataFrame(records)

    def _sample_frames(self, filepath: str) -> torch.Tensor:
        """Returns (num_frames, C, H, W) tensor."""
        video, _, _ = self._tvio.read_video(filepath, pts_unit="sec")
        # video shape: (T, H, W, C)
        total = video.shape[0]
        indices = torch.linspace(0, total - 1, self.num_frames).long()
        frames = video[indices]  # (num_frames, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
        resize = transforms.Resize((self.img_size, self.img_size))
        frames = torch.stack([resize(f) for f in frames])  # (N, C, H, W)
        return frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = self._sample_frames(row["filepath"])  # (N, C, H, W)
        label = int(row["label_idx"])
        return frames, label


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloader(dataset: Dataset, batch_size: int = 32,
                   num_workers: int = 2, balanced: bool = False) -> DataLoader:
    """
    Returns a DataLoader. Set balanced=True to use WeightedRandomSampler
    for imbalanced class distributions.
    """
    if balanced and hasattr(dataset, "get_class_weights"):
        sampler = WeightedRandomSampler(
            weights=dataset.get_class_weights(),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers,
                          pin_memory=True)
    shuffle = (dataset.split == "train") if hasattr(dataset, "split") else True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
