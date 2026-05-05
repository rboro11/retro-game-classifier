"""
build_dataset.py — plug-and-play dataset builder

Run this ONCE after dropping raw files into data/raw/<ClassName>/

Usage:
    python scripts/build_dataset.py --mode frames    # extract frames from MP4s
    python scripts/build_dataset.py --mode audio     # extract .wav from MP4s
    python scripts/build_dataset.py --mode spectrograms  # generate mel-specs
    python scripts/build_dataset.py --mode splits    # create train/val/test CSVs
    python scripts/build_dataset.py --mode all       # run everything
"""

import os
import sys
import shutil
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
# Frame extraction (ffmpeg)
# ─────────────────────────────────────────────

def extract_frames(fps: float = 1.0):
    """
    For each class, extract frames from any MP4/video at <fps> fps.
    Also copies any raw images directly.
    Output: data/processed/frames/<ClassName>/<file>_f<N>.png
    """
    print(f"\n[extract_frames] FPS = {fps}")
    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        out_dir = FRAMES_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Direct images → copy
        for img_file in class_dir.rglob("*"):
            if img_file.suffix.lower() in IMG_EXTS:
                dest = out_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)

        # Videos → ffmpeg frame extraction
        for vid_file in class_dir.rglob("*"):
            if vid_file.suffix.lower() not in VIDEO_EXTS:
                continue
            stem   = vid_file.stem
            output = str(out_dir / f"{stem}_f%05d.png")
            cmd = [
                "ffmpeg", "-y", "-i", str(vid_file),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                output,
                "-loglevel", "error",
            ]
            print(f"  Extracting {vid_file.name} → {out_dir.name}/")
            subprocess.run(cmd, check=True)

    total = sum(1 for _ in FRAMES_DIR.rglob("*.png"))
    print(f"  Done. Total frames: {total:,}")


# ─────────────────────────────────────────────
# Audio extraction (ffmpeg)
# ─────────────────────────────────────────────

def extract_audio(clip_duration: float = 10.0):
    """
    Extract audio clips from MP4s at fixed clip_duration intervals.
    Also copies any raw audio files.
    Output: data/processed/audio/<ClassName>/<file>_clip<N>.wav
    """
    print(f"\n[extract_audio] Clip duration = {clip_duration}s")
    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        out_dir = AUDIO_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Direct audio files → copy/convert
        for aud_file in class_dir.glob("*"):
            if aud_file.suffix.lower() in AUDIO_EXTS:
                dest = out_dir / (aud_file.stem + ".wav")
                if not dest.exists():
                    cmd = ["ffmpeg", "-y", "-i", str(aud_file),
                           "-ar", "22050", "-ac", "1",
                           str(dest), "-loglevel", "error"]
                    subprocess.run(cmd, check=True)

        # Videos → extract audio in chunks
        for vid_file in class_dir.rglob("*"):
            if vid_file.suffix.lower() not in VIDEO_EXTS:
                continue
            # Get duration
            dur_cmd = ["ffprobe", "-v", "error",
                       "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1",
                       str(vid_file)]
            try:
                result = subprocess.run(dur_cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip())
            except Exception:
                duration = 60.0  # assume 1 min if probe fails

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
# Mel-spectrogram PNG generation (optional pre-compute)
# ─────────────────────────────────────────────

def generate_spectrograms():
    """
    Pre-compute mel-spectrograms as PNG images so the audio dataset class
    can work without on-the-fly computation.
    Requires: librosa, PIL
    """
    try:
        import librosa
        import librosa.display
        import numpy as np
        from PIL import Image
    except ImportError:
        print("Install librosa for spectrogram generation: pip install librosa")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
                    y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                # Save as grayscale PNG
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

def build_splits(val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42):
    """
    Scan data/processed/frames/ and data/processed/audio/ to build
    train.csv / val.csv / test.csv.

    Each row: filepath, label, label_idx, modality
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    records = []

    # Image frames
    if FRAMES_DIR.exists():
        for class_dir in sorted(FRAMES_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = [f for f in class_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
            for f in imgs:
                records.append({"filepath": str(f), "label": class_dir.name,
                                "modality": "image"})

    # Audio (only if processed audio directory exists)
    if AUDIO_DIR.exists():
        for class_dir in sorted(AUDIO_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            wavs = list(class_dir.glob("*.wav"))
            for f in wavs:
                records.append({
                    "filepath": str(f),
                    "label": class_dir.name,
                    "modality": "audio"
                })

    if not records:
        print("No processed data found. Run --mode frames or --mode audio first.")
        return

    df = pd.DataFrame(records)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df["label"].map(class_to_idx)

    # Stratified split per class
    train_rows, val_rows, test_rows = [], [], []
    for cls in classes:
        cls_df = df[df["label"] == cls].sample(frac=1, random_state=seed)
        n = len(cls_df)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_rows.append(cls_df.iloc[:n_test])
        val_rows.append(cls_df.iloc[n_test:n_test + n_val])
        train_rows.append(cls_df.iloc[n_test + n_val:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df   = pd.concat(val_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv",     index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv",   index=False)

    # Save class mapping
    pd.DataFrame({"label": classes, "label_idx": range(len(classes))}).to_csv(
        SPLITS_DIR / "classes.csv", index=False
    )

    print(f"\n[build_splits] Done.")
    print(f"  Classes ({len(classes)}): {classes}")
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retro Game Classifier — Dataset Builder")
    parser.add_argument("--mode", choices=["frames", "audio", "spectrograms",
                                           "splits", "all"], default="all")
    parser.add_argument("--fps",  type=float, default=1.0,
                        help="Frames per second for frame extraction")
    parser.add_argument("--clip_duration", type=float, default=10.0,
                        help="Audio clip duration in seconds")
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    mode = args.mode
    if mode in ("frames",       "all"): extract_frames(fps=args.fps)
    if mode in ("audio",        "all"): extract_audio(clip_duration=args.clip_duration)
    if mode in ("spectrograms", "all"): generate_spectrograms()
    if mode in ("splits",       "all"): build_splits(args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
