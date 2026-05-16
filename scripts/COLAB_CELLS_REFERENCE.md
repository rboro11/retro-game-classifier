# Colab Session Reference

Quick reference for all key notebook cells used in this project. Run these at the start of each session or as needed.

---

## 1. Setup Cell (Run First Every Session)

Mounts Drive, clones or pulls the repo, installs dependencies, and creates required directories.

```python
import os, subprocess
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive', force_remount=False)

# 2. Clone or pull repo
repo_path = '/content/retro-game-classifier'
token = __import__('getpass').getpass("GitHub token: ")

if os.path.exists(f"{repo_path}/.git"):
    subprocess.run(["git", "-C", repo_path, "pull", "origin", "main"], check=True)
    print("✅ Repo pulled latest.")
elif os.path.exists(repo_path):
    subprocess.run(["rm", "-rf", repo_path], check=True)
    subprocess.run(["git", "clone",
        f"https://rboro11:{token}@github.com/rboro11/retro-game-classifier.git",
        repo_path], check=True)
    print("✅ Repo re-cloned (stale directory removed).")
else:
    subprocess.run(["git", "clone",
        f"https://rboro11:{token}@github.com/rboro11/retro-game-classifier.git",
        repo_path], check=True)
    print("✅ Repo cloned fresh.")

# 3. Install dependencies
subprocess.run(["pip", "install", "-q", "-r",
    f"{repo_path}/requirements.txt"], check=True)
print("✅ Dependencies installed.")

# 4. Create directories
os.makedirs(f"{repo_path}/data/raw/SMB1", exist_ok=True)
os.makedirs(f"{repo_path}/data/raw/SMB3", exist_ok=True)
os.makedirs("/content/drive/MyDrive/retro-game-classifier/checkpoints", exist_ok=True)
os.makedirs("/content/drive/MyDrive/retro-game-classifier/reports", exist_ok=True)
print("✅ Directories ready.")

os.chdir(repo_path)
print("✅ Ready.")
```

---

## 2. GPU Check

Verify T4 GPU and FFmpeg hardware acceleration are available.

```python
import torch
import subprocess

gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detected: {gpu_name}")
    try:
        hwaccels = subprocess.check_output(["ffmpeg", "-hwaccels"], stderr=subprocess.STDOUT).decode()
        if "cuda" in hwaccels or "nvdec" in hwaccels:
            print("✅ FFmpeg hardware acceleration supported.")
    except:
        print("⚠️ FFmpeg check failed, but GPU is available for PyTorch.")
else:
    print("❌ No GPU detected. Go to Edit > Notebook settings > Hardware accelerator and select 'T4 GPU'.")
```

---

## 3. Set Paths

Define all project paths. Run after the setup cell.

```python
import os
from pathlib import Path

PROJECT_DIR  = Path("/content/retro-game-classifier")
RAW_DIR      = PROJECT_DIR / "data" / "raw"
SMB1_RAW     = RAW_DIR / "SMB1"
SMB3_RAW     = RAW_DIR / "SMB3"
SMB3_VIDEOS  = RAW_DIR / "SMB3_videos"

SMB3_DRIVE_DIR      = Path("/content/drive/MyDrive/retro-game-classifier-project")
CHECKPOINTS_DRIVE   = Path("/content/drive/MyDrive/retro-game-classifier/checkpoints")
REPORTS_DRIVE       = Path("/content/drive/MyDrive/retro-game-classifier/reports")
```

---

## 4. Copy SMB3 Spliced Videos from Drive

Copy all 7 session videos from Drive into the runtime. Run when runtime is fresh.

```python
import shutil
from pathlib import Path

DRIVE_DIR = Path("/content/drive/MyDrive/SMB3 videos")
DEST_DIR  = Path("/content/retro-game-classifier/data/raw/SMB3_videos")
DEST_DIR.mkdir(parents=True, exist_ok=True)

for i in range(1, 8):
    fname = f"training_data_SMB3_{i}_spliced.mp4"
    shutil.copy2(DRIVE_DIR / fname, DEST_DIR / fname)
    print(f"✅ Copied: {fname}")

print(f"\n✅ All 7 spliced videos ready at {DEST_DIR}")
```

---

## 5. Extract Frames from SMB3 Sessions

Extract frames from each spliced video with session ID encoded in the filename.
Produces files like `SMB3_1_frame00042.png` for clean session-level train/val/test splitting.

```python
import subprocess
from pathlib import Path

SPLICED_DIR = Path("/content/retro-game-classifier/data/raw/SMB3_videos")
OUTPUT_DIR  = Path("/content/retro-game-classifier/data/raw/SMB3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 1  # 1 frame per second — adjust as needed

for i in range(1, 8):
    video_path  = SPLICED_DIR / f"training_data_SMB3_{i}_spliced.mp4"
    session_tag = f"SMB3_{i}"

    print(f"Extracting frames from {video_path.name}...")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={FPS}",
        "-q:v", "2",
        str(OUTPUT_DIR / f"{session_tag}_frame%05d.png"),
        "-hide_banner", "-loglevel", "error"
    ], check=True)

    count = len(list(OUTPUT_DIR.glob(f"{session_tag}_frame*.png")))
    print(f"✅ {session_tag}: {count} frames extracted")

total = len(list(OUTPUT_DIR.glob("*.png")))
print(f"\n✅ Total SMB3 frames: {total}")
```

---

## 6. Session-Level Train/Val/Test Split

Split data by source session (not random frame-level) to prevent data leakage.

```python
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("/content/retro-game-classifier/data/raw/SMB3")

def get_session_id(filename):
    # "SMB3_1_frame00042.png" → "SMB3_1"
    return Path(filename).stem.rsplit("_frame", 1)[0]

frames = list(OUTPUT_DIR.glob("*.png"))

# Group frames by session
session_to_frames = defaultdict(list)
for f in frames:
    session_to_frames[get_session_id(f.name)].append(f)

# Split at session level
sessions = list(session_to_frames.keys())
train_sess, temp_sess = train_test_split(sessions, test_size=0.3, random_state=42)
val_sess, test_sess   = train_test_split(temp_sess, test_size=0.5, random_state=42)

# Expand back to frames
train_frames = [f for s in train_sess for f in session_to_frames[s]]
val_frames   = [f for s in val_sess   for f in session_to_frames[s]]
test_frames  = [f for s in test_sess  for f in session_to_frames[s]]

print(f"Sessions — train: {train_sess}, val: {val_sess}, test: {test_sess}")
print(f"Frames   — train: {len(train_frames)}, val: {len(val_frames)}, test: {len(test_frames)}")
```

---

## 7. End-of-Session Push to GitHub

Save the live notebook from Drive and push all changes to GitHub. Run at the end of every session.

```python
from google.colab import drive, _message
import shutil, subprocess, os
from datetime import datetime
from zoneinfo import ZoneInfo

drive.mount('/content/drive', force_remount=False)
_message.blocking_request('save_notebook', request='', timeout_sec=60)

DRIVE_NB  = "/content/drive/MyDrive/Colab Notebooks/retro_game_classifier.ipynb"
REPO_NB   = "/content/retro-game-classifier/retro_game_classifier.ipynb"
repo      = "/content/retro-game-classifier"
token     = __import__('getpass').getpass("GitHub token: ")
est       = datetime.now(tz=ZoneInfo("America/New_York"))
timestamp = est.strftime('%Y-%m-%d %I:%M %p EST')

shutil.copy2(DRIVE_NB, REPO_NB)
print(f"✅ Drive backup saved: {est.strftime('%Y-%m-%d %I:%M:%S %p EST')}")

subprocess.run(["git","-C",repo,"remote","set-url","origin",
    f"https://rboro11:{token}@github.com/rboro11/retro-game-classifier.git"],check=True)
subprocess.run(["git","-C",repo,"config","user.email","rrb24116@gmail.com"],check=True)
subprocess.run(["git","-C",repo,"config","user.name","Ryan Boro"],check=True)
subprocess.run(["git","-C",repo,"add","."],check=True)

r = subprocess.run(["git","-C",repo,"commit","-m",f"Session update - {timestamp}"],
    capture_output=True,text=True)
print(r.stdout or "Nothing new to commit.")

p = subprocess.run(["git","-C",repo,"push","origin","main"],capture_output=True,text=True)
print(p.stdout or p.stderr)
print("✅ GitHub push complete." if p.returncode == 0 else "❌ Push failed — check token scope.")
```

---

## Notes

- **Token scope**: GitHub token requires `repo` (full) scope. Regenerate at [github.com/settings/tokens](https://github.com/settings/tokens) if push fails with 403.
- **Runtime resets**: Colab runtimes are ephemeral. Re-run cells 1 → 3 → 4 → 5 on each fresh runtime. Data files (videos, frames) must be re-copied from Drive each session.
- **Live notebook**: Always work from the Drive-backed tab (`Colab Notebooks/retro_game_classifier.ipynb`), not the cloned repo copy.
- **chunk_splice_video.py**: For creating new session spliced videos, use `scripts/chunk_splice_video.py` instead of the inline notebook cell.
