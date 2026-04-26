# ItsaMe-Mario-Identifier-
DL models to identify between different Mario titles
[README.md](https://github.com/user-attachments/files/27101497/README.md)

Deep learning system to identify which Nintendo Mario game a screenshot, audio clip, or video clip belongs to.

## Quick Start (Google Colab)

```bash
git clone https://github.com/YOUR_USERNAME/mario-identifier
cd mario-identifier
pip install -r requirements.txt
```

1. Drop raw data into `data/raw/<GameName>/`
2. Run `python scripts/build_dataset.py --mode all`
3. Train: `python scripts/train_model.py --model resnet18 --num_classes 3 --epochs 30`
4. Benchmark: `python scripts/run_benchmark.py`

Or open `Mario_Identifier_Colab.ipynb` directly in Colab.

---

## Project Phases

| Phase | Task | Models | Data Needed |
|---|---|---|---|
| 1 | Binary NES classifier | Custom CNN | SMB1 (public) + SMB3 (your captures) |
| 2 | 3–5 game NES classifier | CNN, ResNet-18, EfficientNet-B0 | SMB1 (public) + SMB2/3 (yours) |
| 3 | 10–20 game multi-era | ResNet-50, EfficientNet-B3, ViT-B/16 | MobyGames API + your captures |
| 4 | Audio classifier | SpectrogramCNN, EfficientNet | NES-MDB (public) + your OST rips |
| 5 | Video classifier | CNN+LSTM, 3D-ResNet | NES-VMDB (public) + your recordings |
| 6 | Multi-modal fusion | Late fusion (average/concat/attention) | Phases 3+4 combined |

---

## Public Datasets

| Dataset | Modality | Link |
|---|---|---|
| SMB1 Gameplay (737k frames, CC-BY-4.0) | Images | [rafaelcp/smbdataset](https://github.com/rafaelcp/smbdataset) |
| NES-MDB (5,278 tracks, 397 games) | Audio/MIDI | [chrisdonahue/nesmdb](https://github.com/chrisdonahue/nesmdb) |
| NES-VMDB (98,940 clips, 389 games) | Video+Audio | [arXiv 2404.04420](https://arxiv.org/abs/2404.04420) |
| SMO Thumbnail DB | Images | [Amethyst-szs/smo-thumbnail-database](https://github.com/Amethyst-szs/smo-thumbnail-database) |
| MobyGames API | Images | [mobygames.com/info/api](https://www.mobygames.com/info/api/) |

---

## Repo Structure

```
mario-identifier/
├── src/
│   ├── data/
│   │   └── dataset.py          # MarioImageDataset, MarioAudioDataset, MarioVideoDataset
│   ├── models/
│   │   ├── cnn_custom.py       # MarioCNNSmall, MarioCNNMedium
│   │   ├── transfer_models.py  # ResNet-18/50, EfficientNet-B0/B3, MobileNet, ViT
│   │   ├── temporal_models.py  # CNN+LSTM, 3D-ResNet, FrameAverageWrapper
│   │   ├── audio_model.py      # SpectrogramCNN, SpectrogramTransferNet
│   │   └── fusion_model.py     # AverageFusion, ConcatFusion, AttentionFusion
│   ├── training/
│   │   └── trainer.py          # Universal Trainer (AMP, early stopping, checkpointing)
│   └── evaluation/
│       └── benchmarker.py      # Multi-model comparison report generator
├── scripts/
│   ├── build_dataset.py        # Frame extraction, audio extraction, splits
│   ├── train_model.py          # Single model training CLI
│   └── run_benchmark.py        # Generate comparison report
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                    # Drop your files here
│   └── processed/              # Auto-generated
├── checkpoints/                # Auto-generated during training
├── reports/                    # Benchmark outputs
├── Mario_Identifier_Colab.ipynb
└── requirements.txt
```

---

## Adding a New Game (Plug-and-Play)

```bash
# 1. Create a folder
mkdir data/raw/SuperMarioGalaxy

# 2. Drop MP4 recordings or PNG screenshots in
cp ~/my_gameplay/*.mp4 data/raw/SuperMarioGalaxy/

# 3. Rebuild dataset
python scripts/build_dataset.py --mode all

# 4. Retrain (new class auto-detected)
python scripts/train_model.py --model resnet18 --num_classes <new_count> --epochs 30

# 5. Re-benchmark
python scripts/run_benchmark.py
```

That's it. No code changes needed.
