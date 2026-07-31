"""
build_dataset.py — plug-and-play dataset builder

Run this ONCE after dropping raw files into data/raw/<ClassName>/

Usage:
    python scripts/build_dataset.py --mode frames
    python scripts/build_dataset.py --mode frames --fps 0.3
    python scripts/build_dataset.py --mode frames --classes SMB2 SMB3
    python scripts/build_dataset.py --mode frames --classes SMB2 --target_frames_per_class 1000
    python scripts/build_dataset.py --mode audio
    python scripts/build_dataset.py --mode spectrograms
    python scripts/build_dataset.py --mode splits
    python scripts/build_dataset.py --mode splits --max_per_class 1000
    python scripts/build_dataset.py --mode all

Core arguments:
    --mode {frames,audio,spectrograms,splits,all}
        Which stage(s) to run.

    --fps FLOAT
        Fixed extraction rate for video-to-frame conversion.
        Example: --fps 0.3 extracts about one frame every 3.33 seconds.

    --classes CLASS [CLASS ...]
        Optional subset of class folders to process.
        Example: --classes SMB2 SMB3

    --target_frames_per_class INT
        Runtime-aware frame extraction target.
        Instead of using one fixed FPS for every class, the script computes
        a per-class FPS from total video runtime so the class ends up with
        about this many extracted frames across its full playthrough(s).
        Example: --classes SMB2 --target_frames_per_class 1000

    --clip_duration FLOAT
        Duration used by audio/spectrogram-related steps where applicable.

    --val_ratio FLOAT
        Validation split ratio.

    --test_ratio FLOAT
        Test split ratio.

    --max_per_class INT
        Cap each class at this many samples during split generation
        (0 = no cap).
        Recommended: set to about the size of your smallest class, or
        slightly above it, to reduce imbalance without throwing away too much.

Disk-usage behavior:
    Raw images (PNG/JPG/etc.) placed in data/raw/<Class>/ are referenced
    directly in the split CSVs and are not copied into data/processed/.
    Only frames extracted from videos are written into:
        data/processed/frames/<Class>/

Why use --target_frames_per_class:
    This is useful when one class is represented by a full gameplay video and
    you want roughly N samples distributed across the entire playthrough.
    Example:
        python scripts/build_dataset.py --mode frames --classes SMB2 --target_frames_per_class 1000
    For a shorter video, the computed FPS will be higher.
    For a longer video, the computed FPS will be lower.

Split strategy notes:
    1. Subdirectory layout:
       If a class is organized into subfolders (for example by level, run,
       or session), the script can split at the subdirectory/session level so
       near-duplicate neighboring frames stay together.

    2. Flat frame layout with _fNNNNN naming:
       If files are named like:
           SMB3_1_cropped_f00042.png
       the script can group by source video stem and split by clip instead of
       by individual frame, which helps prevent leakage across train/val/test.

    3. Single-video fallback:
       If only one source clip exists for a class, clip-level splitting is not
       enough to populate all three splits. In that case, the script can fall
       back to a chronological frame-level split based on filename order:
           test  = early frames
           val   = middle frames
           train = later frames

    4. Generic flat image layout:
       If there are no session folders and no _fNNNNN naming pattern, files
       are split at the individual image level.

Recommended examples:
    Fixed FPS for all available classes:
        python scripts/build_dataset.py --mode frames --fps 0.3

    Runtime-aware extraction for one class:
        python scripts/build_dataset.py --mode frames --classes SMB2 --target_frames_per_class 1000

    Build balanced splits after extraction:
        python scripts/build_dataset.py --mode splits --max_per_class 1000

Typical local-runtime workflow:
    1. Place raw source data under data/raw/<Class>/
    2. Run frame extraction
    3. Run split generation
    4. Train/evaluate/export from the generated CSVs

Notes:
    - If a class contains only images and no videos, --mode frames will skip
      video extraction for that class.
    - If a class contains no detectable video runtime, the script can fall
      back to the value passed via --fps.
    - This workflow aligns with the current Colab notebook pattern that builds
      frames first, then builds splits locally. 
"""

import os
import re
import csv
import math
import random
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FRAMES_DIR = ROOT / "data" / "processed" / "frames"
AUDIO_DIR = ROOT / "data" / "processed" / "audio"
SPECS_DIR = ROOT / "data" / "processed" / "spectrograms"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
FRAME_RE = re.compile(r"_f(\d+)\\.[A-Za-z0-9]+$")


def ensure_dirs():
    for p in [FRAMES_DIR, AUDIO_DIR, SPECS_DIR, SPLITS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def find_video_files(class_dir: Path):
    return sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def compute_runtime_scaled_fps(class_dir: Path, target_frames: int) -> float | None:
    videos = find_video_files(class_dir)
    if not videos:
        return None
    total_seconds = sum(probe_duration_seconds(v) for v in videos)
    if total_seconds <= 0:
        return None
    return max(target_frames / total_seconds, 0.0001)


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


def extract_frames(fps: float = 1.0, classes: list[str] | None = None, runtime_scaled: bool = False, target_frames_per_class: int | None = None):
    if runtime_scaled:
        print(f"\n[extract_frames] Runtime-scaled extraction enabled (target_frames_per_class={target_frames_per_class}; images skipped — referenced directly from raw/)")
    else:
        print("\n[extract_frames] Starting extraction  (images skipped — referenced directly from raw/)")
    hw_args = _ffmpeg_hwaccel_args()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw data directory: {RAW_DIR}")

    class_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if classes:
        wanted = set(classes)
        class_dirs = [p for p in class_dirs if p.name in wanted]

    total = 0
    for class_dir in class_dirs:
        videos = find_video_files(class_dir)
        if not videos:
            print(f"  {class_dir.name}: no videos found, skipping.")
            continue

        out_dir = FRAMES_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for vid_file in videos:
            stem = vid_file.stem
            output = str(out_dir / f"{stem}_f%05d.png")
            print(f"  Extracting {vid_file.name} → {class_dir.name}/ at {fps:.4f} fps")
            cmd = ["ffmpeg", "-y", *hw_args, "-i", str(vid_file), "-vf", f"fps={fps}", output]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            total += len(list(out_dir.glob(f"{stem}_f*.png")))
    print(f"  Done. Total video-extracted frames: {total}")


def _video_stem(path: Path) -> str:
    m = FRAME_RE.search(path.name)
    if m:
        return path.name[:m.start()]
    return path.stem


def _frame_index(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _session_level_split(files, val_ratio=0.15, test_ratio=0.15, seed=42):
    groups = defaultdict(list)
    for p in files:
        rel = p.parent.name
        groups[rel].append(p)
    sessions = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n = len(sessions)
    n_test = max(1, round(n * test_ratio)) if n >= 3 else max(0, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio)) if n >= 3 else max(0, round(n * val_ratio))
    test_sessions = set(sessions[:n_test])
    val_sessions = set(sessions[n_test:n_test + n_val])
    train_sessions = set(sessions[n_test + n_val:])
    train = [p for s in train_sessions for p in groups[s]]
    val = [p for s in val_sessions for p in groups[s]]
    test = [p for s in test_sessions for p in groups[s]]
    return train, val, test


def _video_stem_split(files, val_ratio=0.15, test_ratio=0.15, seed=42):
    groups = defaultdict(list)
    for p in files:
        groups[_video_stem(p)].append(p)
    stems = list(groups.keys())

    if len(stems) < 3:
        ordered = sorted(files, key=lambda p: (_video_stem(p), _frame_index(p), p.name))
        n = len(ordered)
        n_test = max(1, round(n * test_ratio)) if n >= 3 else max(0, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio)) if n >= 3 else max(0, round(n * val_ratio))
        test = ordered[:n_test]
        val = ordered[n_test:n_test + n_val]
        train = ordered[n_test + n_val:]
        print("  WARNING: only", len(stems), "video clips detected; falling back to chronological frame-level split.")
        return train, val, test

    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    test_stems = set(stems[:n_test])
    val_stems = set(stems[n_test:n_test + n_val])
    train_stems = set(stems[n_test + n_val:])
    train = [p for s in train_stems for p in groups[s]]
    val = [p for s in val_stems for p in groups[s]]
    test = [p for s in test_stems for p in groups[s]]
    return train, val, test


def _generic_file_split(files, val_ratio=0.15, test_ratio=0.15, seed=42):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    return train, val, test


def build_splits(val_ratio=0.15, test_ratio=0.15, max_per_class=0, seed=42):
    ensure_dirs()
    classes = []
    train_rows, val_rows, test_rows = [], [], []

    if RAW_DIR.exists():
        raw_classes = {p.name for p in RAW_DIR.iterdir() if p.is_dir()}
    else:
        raw_classes = set()
    frame_classes = {p.name for p in FRAMES_DIR.iterdir() if p.is_dir()} if FRAMES_DIR.exists() else set()
    class_names = sorted(raw_classes | frame_classes)

    for label_idx, cls in enumerate(class_names):
        raw_cls = RAW_DIR / cls
        frame_cls = FRAMES_DIR / cls
        files = []
        if frame_cls.exists():
            files.extend([p for p in frame_cls.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if raw_cls.exists():
            files.extend([p for p in raw_cls.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        files = sorted(set(files))
        if not files:
            continue

        classes.append((cls, label_idx))

        subdirs = {p.parent.relative_to(raw_cls).parts[0] for p in files if raw_cls.exists() and p.is_relative_to(raw_cls) and len(p.parent.relative_to(raw_cls).parts) > 0}
        has_sessions = len(subdirs) > 1
        has_frame_naming = any(FRAME_RE.search(p.name) for p in files)

        if has_sessions:
            train, val, test = _session_level_split(files, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        elif has_frame_naming:
            train, val, test = _video_stem_split(files, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        else:
            train, val, test = _generic_file_split(files, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

        if max_per_class and max_per_class > 0:
            rng = random.Random(seed)
            def cap(xs):
                xs = list(xs)
                if len(xs) <= max_per_class:
                    return xs
                rng.shuffle(xs)
                return xs[:max_per_class]
            train, val, test = cap(train), cap(val), cap(test)

        for split_name, rows, xs in [("train", train_rows, train), ("val", val_rows, val), ("test", test_rows, test)]:
            for p in xs:
                rows.append({"filepath": str(p), "label": cls, "label_idx": label_idx, "split": split_name})

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(classes, columns=["label", "label_idx"]).to_csv(SPLITS_DIR / "classes.csv", index=False)
    pd.DataFrame(train_rows).to_csv(SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(SPLITS_DIR / "val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(SPLITS_DIR / "test.csv", index=False)
    print("[build_splits] Done.")
    print("  Classes:", len(classes), ", ".join([c for c, _ in classes]))
    print("  Train:", len(train_rows), "Val:", len(val_rows), "Test:", len(test_rows))


def parse_args():
    ap = argparse.ArgumentParser(description="Retro Game Classifier — Dataset Builder")
    ap.add_argument("--mode", default="all", choices=["frames", "audio", "spectrograms", "splits", "all"])
    ap.add_argument("--fps", type=float, default=0.3)
    ap.add_argument("--clip_duration", type=float, default=3.0)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Cap each class at this many samples (0 = no cap). Recommended: set to ~2x your smallest class size to reduce imbalance.")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Optional subset of class folders to process, e.g. --classes SMB2 SMB3")
    ap.add_argument("--target_frames_per_class", type=int, default=None,
                    help="If set for --mode frames, compute fps per class from total runtime to target this many frames across the class")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    if args.mode in ["frames", "all"]:
        if args.classes and args.target_frames_per_class:
            for cls in args.classes:
                class_dir = RAW_DIR / cls
                fps = compute_runtime_scaled_fps(class_dir, args.target_frames_per_class)
                if fps is None:
                    fps = args.fps
                    print(f"[frames] {cls}: could not compute runtime-scaled fps, falling back to --fps {fps}")
                else:
                    print(f"[frames] {cls}: runtime-scaled fps = {fps:.6f} for target {args.target_frames_per_class}")
                extract_frames(fps=fps, classes=[cls], runtime_scaled=True, target_frames_per_class=args.target_frames_per_class)
        else:
            extract_frames(fps=args.fps, classes=args.classes)

    if args.mode in ["audio", "all"]:
        print("[audio] mode selected, but audio extraction is not implemented in this script yet.")

    if args.mode in ["spectrograms", "all"]:
        print("[spectrograms] mode selected, but spectrogram generation is not implemented in this script yet.")

    if args.mode in ["splits", "all"]:
        build_splits(
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_per_class=args.max_per_class,
        )


if __name__ == "__main__":
    main()

