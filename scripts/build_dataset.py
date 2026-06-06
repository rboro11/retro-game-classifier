"""
build_dataset.py — plug-and-play dataset builder

Run this ONCE after dropping raw files into data/raw/<ClassName>/

Usage:
    python scripts/build_dataset.py --mode frames    # extract frames from MP4s only (no image copy)
    python scripts/build_dataset.py --mode audio     # extract .wav from MP4s
    python scripts/build_dataset.py --mode spectrograms  # generate mel-specs
    python scripts/build_dataset.py --mode splits    # create train/val/test CSVs
    python scripts/build_dataset.py --mode all       # run everything

NOTE on disk usage:
    Raw images (PNG/JPG) in data/raw/<Class>/ are referenced DIRECTLY in the
    split CSVs — they are never copied into data/processed/.
    Only video-extracted frames land in data/processed/frames/.

NOTE on class imbalance:
    Use --max_per_class N to cap any class at N samples (random, seeded).
    Example: python scripts/build_dataset.py --mode splits --max_per_class 5000
    Also pass --weighted_loss to print the loss weights for use in training.
"""

import os
import argparse
import subprocess
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
FRAMES_DIR = ROOT / "data" / "processed" / "frames"
AUDIO_DIR  = ROOT / "data" / "processed" / "audio"
SPECS_DIR  = ROOT / "data" / "processed" / "spectrograms"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

IMG_EXTS   = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


# ─────────────────────────────────────────────
# GPU detection helper
# ─────────────────────────────────────────────

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ffmpeg_hwaccel_args() -> list:
    if _gpu_available():
        print("  [GPU detected] Using -hwaccel auto for frame extraction.")
        return ["-hwaccel", "auto"]
    print("  [No GPU] Using CPU for frame extraction.")
    return []


# ─────────────────────────────────────────────
# Frame extraction (ffmpeg — videos only)
# ─────────────────────────────────────────────

def extract_frames(fps: float = 1.0):
    """
    Extract frames from video files only. Raw images are NOT copied.
    Output: data/processed/frames/<ClassName>/<videoname>_f<N>.png
    """
    print(f"\n[extract_frames] FPS = {fps}  (images skipped — referenced directly from raw/)")
    hw_args = _ffmpeg_hwaccel_args()

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        videos = [f for f in class_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTS]
        if not videos:
            print(f"  {class_dir.name}: no videos found, skipping.")
            continue

        out_dir = FRAMES_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for vid_file in videos:
            stem = vid_file.stem
            output = str(out_dir / f"{stem}_f%05d.png")

            existing = list(out_dir.glob(f"{stem}_f*.png"))
            if existing:
                print(f"  Skipping {vid_file.name} ({len(existing)} frames already exist)")
                continue

            cmd = (
                ["ffmpeg", "-y"]
                + hw_args
                + ["-i", str(vid_file), "-vf", f"fps={fps}",
                   "-q:v", "2", output, "-hide_banner", "-loglevel", "error"]
            )

            print(f"  Extracting {vid_file.name} → {out_dir.name}/")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                if hw_args:
                    print(f"  Warning: hwaccel failed for {vid_file.name}, retrying on CPU...")
                    cmd_cpu = (
                        ["ffmpeg", "-y"]
                        + ["-i", str(vid_file), "-vf", f"fps={fps}",
                           "-q:v", "2", output, "-hide_banner", "-loglevel", "error"]
                    )
                    try:
                        subprocess.run(cmd_cpu, check=True)
                    except subprocess.CalledProcessError as e2:
                        print(f"  ERROR: {vid_file.name}: {e2}")
                else:
                    print(f"  ERROR: Could not extract frames from {vid_file.name}")

    total = sum(1 for _ in FRAMES_DIR.rglob("*.png")) if FRAMES_DIR.exists() else 0
    print(f"  Done. Total video-extracted frames: {total:,}")


# ─────────────────────────────────────────────
# Audio extraction (ffmpeg)
# ─────────────────────────────────────────────

def extract_audio(clip_duration: float = 10.0):
    """
    Extract audio clips from MP4s at fixed clip_duration intervals.
    Output: data/processed/audio/<ClassName>/<file>_clip<N>.wav
    """
    print(f"\n[extract_audio] Clip duration = {clip_duration}s")
    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        out_dir = AUDIO_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for aud_file in class_dir.glob("*"):
            if aud_file.suffix.lower() in AUDIO_EXTS:
                dest = out_dir / (aud_file.stem + ".wav")
                if not dest.exists():
                    cmd = ["ffmpeg", "-y", "-i", str(aud_file),
                           "-ar", "22050", "-ac", "1",
                           str(dest), "-loglevel", "error"]
                    subprocess.run(cmd, check=True)

        for vid_file in class_dir.rglob("*"):
            if vid_file.suffix.lower() not in VIDEO_EXTS:
                continue
            dur_cmd = ["ffprobe", "-v", "error",
                       "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1",
                       str(vid_file)]
            try:
                result = subprocess.run(dur_cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip())
            except Exception:
                duration = 60.0

            n_clips = max(1, int(duration / clip_duration))
            for i in range(n_clips):
                start = i * clip_duration
                out_path = out_dir / f"{vid_file.stem}_clip{i:04d}.wav"
                if out_path.exists():
                    continue
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-i", str(vid_file),
                    "-t", str(clip_duration),
                    "-ar", "22050", "-ac", "1",
                    str(out_path), "-loglevel", "error",
                ]
                subprocess.run(cmd, check=True)
            print(f"  {vid_file.name}: {n_clips} clips → {out_dir.name}/")

    total = sum(1 for _ in AUDIO_DIR.rglob("*.wav"))
    print(f"  Done. Total audio clips: {total:,}")


# ─────────────────────────────────────────────
# Mel-spectrogram PNG generation
# ─────────────────────────────────────────────

def generate_spectrograms():
    try:
        import librosa
        import numpy as np
        from PIL import Image
    except ImportError:
        print("Install librosa: pip install librosa")
        return

    import matplotlib
    matplotlib.use("Agg")

    print("\n[generate_spectrograms]")
    if not AUDIO_DIR.exists():
        print("  No audio directory found. Run --mode audio first.")
        return

    for class_dir in sorted(AUDIO_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        out_dir = SPECS_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for wav_file in tqdm(list(class_dir.glob("*.wav")),
                             desc=class_dir.name, leave=False):
            dest = out_dir / (wav_file.stem + ".png")
            if dest.exists():
                continue
            try:
                y, sr = librosa.load(str(wav_file), sr=22050, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = ((mel_db - mel_db.min()) /
                            (mel_db.max() - mel_db.min() + 1e-8) * 255).astype("uint8")
                Image.fromarray(mel_norm).save(dest)
            except Exception as e:
                print(f"  Warning: could not process {wav_file.name}: {e}")

    total = sum(1 for _ in SPECS_DIR.rglob("*.png"))
    print(f"  Done. Total spectrograms: {total:,}")


# ─────────────────────────────────────────────
# Train / Val / Test splits
# ─────────────────────────────────────────────

def _collect_image_records(class_dir: Path, label: str) -> list:
    """Return a list of image record dicts from a single class directory."""
    return [
        {"filepath": str(f), "label": label, "modality": "image"}
        for f in class_dir.rglob("*")
        if f.suffix.lower() in IMG_EXTS
    ]


def build_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_per_class: int = 0,   # 0 = no cap
):
    """
    Build train.csv / val.csv / test.csv.

    Source priority per class (no double-counting):
      • If data/processed/frames/<Class>/ exists  → use processed frames
        (these came from videos; raw/ has no images for this class)
      • Otherwise                                 → use data/raw/<Class>/ directly
        (image-only class; never copied to processed/)

    max_per_class: if > 0, randomly subsample each class to this many files.
    This is the primary tool for fixing class imbalance.

    Each CSV row: filepath, label, label_idx, modality
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    records = []

    # Gather all known class names from both raw/ and processed/frames/
    raw_classes    = {d.name for d in RAW_DIR.iterdir()    if d.is_dir()} if RAW_DIR.exists()    else set()
    frames_classes = {d.name for d in FRAMES_DIR.iterdir() if d.is_dir()} if FRAMES_DIR.exists() else set()
    all_classes    = raw_classes | frames_classes

    for cls in sorted(all_classes):
        frames_dir = FRAMES_DIR / cls
        raw_dir    = RAW_DIR    / cls

        if frames_dir.exists() and any(frames_dir.rglob("*")):
            # Class has video-extracted frames — use processed/frames/ only
            cls_records = _collect_image_records(frames_dir, cls)
            source = "video frames"
        elif raw_dir.exists():
            # Image-only class — reference raw/ directly
            cls_records = _collect_image_records(raw_dir, cls)
            source = "raw images"
        else:
            print(f"  WARNING: no data found for class '{cls}', skipping.")
            continue

        n_raw = len(cls_records)

        # Apply per-class cap if requested
        if max_per_class > 0 and n_raw > max_per_class:
            cls_records = random.sample(cls_records, max_per_class)
            print(f"  [{source}] {cls}: {n_raw:,} → capped at {max_per_class:,}")
        else:
            print(f"  [{source}] {cls}: {n_raw:,} files")

        records.extend(cls_records)

    # --- Audio clips (always additive, no overlap with images) ---
    if AUDIO_DIR.exists():
        for class_dir in sorted(AUDIO_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            wavs = list(class_dir.glob("*.wav"))
            for f in wavs:
                records.append({
                    "filepath": str(f),
                    "label": class_dir.name,
                    "modality": "audio",
                })
            if wavs:
                print(f"  [audio] {class_dir.name}: {len(wavs):,} clips")

    if not records:
        print("No data found. Check data/raw/ and data/processed/.")
        return

    df = pd.DataFrame(records)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df["label"].map(class_to_idx)

    # Print class distribution + suggested loss weights
    print("\n  Class distribution:")
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    for cls, n in counts.items():
        print(f"    {cls}: {n:,}  ({n/total*100:.1f}%)")

    # Weighted loss hint (inverse-frequency)
    max_count = counts.max()
    weights   = {cls: round(max_count / n, 4) for cls, n in counts.items()}
    print(f"\n  Suggested CrossEntropyLoss weights (inverse-frequency):")
    print(f"    {weights}")
    print(f"  Usage: loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([{', '.join(str(weights[c]) for c in sorted(weights))}]))")

    # Stratified split per class
    train_rows, val_rows, test_rows = [], [], []
    for cls in classes:
        cls_df = df[df["label"] == cls].sample(frac=1, random_state=seed)
        n = len(cls_df)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_rows.append(cls_df.iloc[:n_test])
        val_rows.append( cls_df.iloc[n_test:n_test + n_val])
        train_rows.append(cls_df.iloc[n_test + n_val:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df   = pd.concat(val_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    test_df.to_csv( SPLITS_DIR / "test.csv",  index=False)

    pd.DataFrame({"label": classes, "label_idx": range(len(classes))}).to_csv(
        SPLITS_DIR / "classes.csv", index=False
    )

    # Save weights for training scripts to load
    pd.DataFrame([
        {"label": cls, "label_idx": class_to_idx[cls], "loss_weight": weights[cls]}
        for cls in classes
    ]).to_csv(SPLITS_DIR / "class_weights.csv", index=False)
    print(f"  Loss weights saved to: data/processed/splits/class_weights.csv")

    print(f"\n[build_splits] Done.")
    print(f"  Classes ({len(classes)}): {classes}")
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print(f"  Total unique files indexed: {len(df):,}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retro Game Classifier — Dataset Builder")
    parser.add_argument("--mode", choices=["frames", "audio", "spectrograms",
                                           "splits", "all"], default="all")
    parser.add_argument("--fps",           type=float, default=1.0)
    parser.add_argument("--clip_duration", type=float, default=10.0)
    parser.add_argument("--val_ratio",     type=float, default=0.15)
    parser.add_argument("--test_ratio",    type=float, default=0.15)
    parser.add_argument(
        "--max_per_class", type=int, default=0,
        help="Cap each class at this many samples (0 = no cap). "
             "Recommended: set to ~2x your smallest class size to reduce imbalance."
    )
    args = parser.parse_args()

    mode = args.mode
    if mode in ("frames",       "all"): extract_frames(fps=args.fps)
    if mode in ("audio",        "all"): extract_audio(clip_duration=args.clip_duration)
    if mode in ("spectrograms", "all"): generate_spectrograms()
    if mode in ("splits",       "all"):
        build_splits(
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_per_class=args.max_per_class,
        )


if __name__ == "__main__":
    main()
