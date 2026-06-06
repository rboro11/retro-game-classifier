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
    split CSVs — they are never copied into data/processed/.  Only video-extracted
    frames land in data/processed/frames/.  This avoids doubling storage for
    large image-only datasets like SMB1.
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
    """Return True if a CUDA-capable GPU is visible to PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ffmpeg_hwaccel_args() -> list:
    """
    Return hardware-acceleration prefix args for ffmpeg if a GPU is available.
    '-hwaccel auto' lets ffmpeg pick the best decoder (CUDA/NVDEC on NVIDIA,
    VAAPI on Linux/Intel, VideoToolbox on macOS).  Falls back to CPU silently
    if the codec or platform doesn't support it.
    """
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
    For each class folder under data/raw/, extract frames from video files
    only (MP4, MOV, AVI, MKV) at <fps> fps.

    Raw images are NOT copied — build_splits() references them directly
    from data/raw/ to avoid doubling disk usage.

    Output: data/processed/frames/<ClassName>/<videoname>_f<N>.png

    GPU acceleration is used automatically when a CUDA GPU is available
    via '-hwaccel auto' prepended to the ffmpeg command.
    """
    print(f"\n[extract_frames] FPS = {fps}  (images skipped — referenced directly from raw/)")
    hw_args = _ffmpeg_hwaccel_args()

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        # Count videos to process
        videos = [f for f in class_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTS]
        if not videos:
            print(f"  {class_dir.name}: no videos found, skipping.")
            continue

        out_dir = FRAMES_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for vid_file in videos:
            stem = vid_file.stem
            output = str(out_dir / f"{stem}_f%05d.png")

            # Skip if frames were already extracted for this video
            existing = list(out_dir.glob(f"{stem}_f*.png"))
            if existing:
                print(f"  Skipping {vid_file.name} ({len(existing)} frames already exist)")
                continue

            # hwaccel args go BEFORE -i; other args after
            cmd = (
                ["ffmpeg", "-y"]
                + hw_args
                + [
                    "-i", str(vid_file),
                    "-vf", f"fps={fps}",
                    "-q:v", "2",
                    output,
                    "-hide_banner",
                    "-loglevel", "error",
                ]
            )

            print(f"  Extracting {vid_file.name} → {out_dir.name}/")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                # If hwaccel caused a failure, retry without it
                if hw_args:
                    print(f"  Warning: hwaccel failed for {vid_file.name}, retrying on CPU...")
                    cmd_cpu = (
                        ["ffmpeg", "-y"]
                        + [
                            "-i", str(vid_file),
                            "-vf", f"fps={fps}",
                            "-q:v", "2",
                            output,
                            "-hide_banner",
                            "-loglevel", "error",
                        ]
                    )
                    try:
                        subprocess.run(cmd_cpu, check=True)
                    except subprocess.CalledProcessError as e2:
                        print(f"  ERROR: Could not extract frames from {vid_file.name}: {e2}")
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
    Build train.csv / val.csv / test.csv by scanning:

      1. data/raw/<Class>/          — raw images referenced directly (no copy)
      2. data/processed/frames/<Class>/ — frames extracted from videos
      3. data/processed/audio/<Class>/  — audio clips (if present)

    Raw images are listed with their original path so no duplication occurs.
    Video-extracted frames are listed from processed/frames/.

    Each row: filepath, label, label_idx, modality
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    records = []

    # --- Raw images (referenced in-place, no copy) ---
    if RAW_DIR.exists():
        for class_dir in sorted(RAW_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = [f for f in class_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
            for f in imgs:
                records.append({
                    "filepath": str(f),
                    "label": class_dir.name,
                    "modality": "image",
                })
            if imgs:
                print(f"  [raw images] {class_dir.name}: {len(imgs):,} files")

    # --- Video-extracted frames ---
    if FRAMES_DIR.exists():
        for class_dir in sorted(FRAMES_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = [f for f in class_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
            for f in imgs:
                records.append({
                    "filepath": str(f),
                    "label": class_dir.name,
                    "modality": "image",
                })
            if imgs:
                print(f"  [video frames] {class_dir.name}: {len(imgs):,} files")

    # --- Audio clips ---
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

    # De-duplicate: same file path could theoretically appear twice
    df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

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

    pd.DataFrame({"label": classes, "label_idx": range(len(classes))}).to_csv(
        SPLITS_DIR / "classes.csv", index=False
    )

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
    parser.add_argument("--fps",  type=float, default=1.0,
                        help="Frames per second for frame extraction (videos only)")
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
