# Retro Game Classifier

A deep learning research project for classifying retro game screenshots, audio clips, and short video clips using public and self-collected datasets.

## Purpose
This repository is intended as a portfolio and research project focused on computer vision, audio classification, experiment design, and model benchmarking. It is not affiliated with, endorsed by, or sponsored by Nintendo or any other game publisher. Nintendo names, characters, game titles, and related assets remain the property of their respective rights holders.

## Scope
The goal is to build a plug-and-play classification pipeline that can:
- identify a game title from a screenshot,
- identify a game title from an audio clip,
- identify a game title from a short video clip,
- compare model performance across architectures and modalities.

The system is designed so that new classes can be added by dropping data into a new folder and rebuilding the dataset, without changing core model code.

---

## Phase 1 Results — Binary Visual Classifier (SMB1 vs SMB3)

Phase 1 trains and benchmarks three architectures on a binary screenshot classification task: Super Mario Bros. (SMB1) vs. Super Mario Bros. 3 (SMB3).

### Test Set
- **1,489 held-out frames** from two SMB3 clips (`SMB3_2_cropped`, `SMB3_4_cropped`) never seen during training
- Split is video-clip-level — no frames from the same source clip appear in both train and test
- Evaluated with `scripts/evaluate.py` against `data/processed/splits/test.csv`

### Benchmark Results

| Model | Test Accuracy | Macro F1 | Precision | Recall | Train Time |
|---|---|---|---|---|---|
| EfficientNet-B0 | **100.00%** | 1.0000 | 1.0000 | 1.0000 | 8.7 min |
| ResNet-18 | **99.80%** | 0.9980 | 0.9980 | 0.9980 | 15.8 min |
| Custom CNN (scratch) | **87.98%** | 0.8779 | 0.9025 | 0.8791 | 4.3 min |

*All models trained on a single Colab T4 GPU. Transfer models fine-tuned from ImageNet weights.*

### Per-Class Breakdown — Custom CNN

The custom CNN (trained from scratch with no pretrained weights) exposes the real difficulty of the task without ImageNet priors:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SMB1 | 1.00 | 0.76 | 0.86 | 740 |
| SMB3 | 0.81 | 1.00 | 0.89 | 749 |

SMB1 recall at 76% reflects the model defaulting to SMB3 when uncertain — a known behavior of small from-scratch models on visually asymmetric classes. EfficientNet-B0 and ResNet-18 achieve perfect or near-perfect scores on both classes because ImageNet pretrained features transfer almost trivially to the distinct color palettes and tile art separating these two games.

### Key Takeaway
Transfer learning delivers a 12-point accuracy gain (88% → 100%) over a custom CNN trained from scratch, with EfficientNet-B0 converging at epoch 1 in under 9 minutes — demonstrating that pretrained visual representations are highly effective even for narrow retro game classification tasks.

---

## Current Focus
The recommended first milestone is a binary screenshot classifier using two visually distinct retro platformer classes. This phase is intended to validate the full pipeline:
- data ingestion,
- preprocessing and split generation,
- model training,
- evaluation and benchmarking,
- reproducible experiment structure.

## Quick Start (Google Colab)
```bash
git clone https://github.com/rboro11/retro-game-classifier
cd retro-game-classifier
pip install -r requirements.txt
```

1. Place raw data into `data/raw/<GameName>/`
2. Run `python scripts/build_dataset.py --mode all`
3. Train a model: `python scripts/train_model.py --model resnet18 --num_classes 2 --epochs 30`
4. Benchmark all models: `python scripts/run_benchmark.py`
5. Evaluate on the held-out test set: `python scripts/evaluate.py`

Or open the project notebook directly in Colab.

### Evaluate a single model
```bash
python scripts/evaluate.py --model cnn_small
python scripts/evaluate.py --model ResNet-18
python scripts/evaluate.py --model EfficientNet-B0
```

Outputs a full classification report (accuracy, precision, recall, F1, support per class) and saves confusion matrix PNGs to `reports/`.

## Project Phases

| Phase | Task | Models | Data Needed |
|---|---|---|---|
| 1 ✅ | Binary visual classifier | Custom CNN, ResNet-18, EfficientNet-B0 | Self-collected SMB1 + SMB3 gameplay captures |
| 2 | 3-5 game visual classifier | CNN, ResNet-18, EfficientNet-B0 | Public and self-collected retro gameplay screenshots |
| 3 | 10-20 game multi-era classifier | ResNet-50, EfficientNet-B3, ViT-B/16 | Public screenshot datasets, APIs, and self-collected captures |
| 4 | Audio classifier | Spectrogram CNN, transfer models | Public audio datasets and self-collected audio clips |
| 5 | Video classifier | CNN+LSTM, 3D-ResNet | Public gameplay clips and self-collected recordings |
| 6 | Multi-modal fusion | Late fusion, concat, attention | Combined image, audio, and optional video data |

## Public Datasets
The project may use public or research-available sources such as:
- gameplay frame datasets,
- audio or MIDI datasets,
- public video datasets,
- screenshot APIs,
- self-collected gameplay captures.

Each dataset should be documented with its source, license, and any usage restrictions before use in training or redistribution.

## Repository Structure
```text
retro-game-classifier/
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── cnn_custom.py
│   │   ├── transfer_models.py
│   │   ├── temporal_models.py
│   │   ├── audio_model.py
│   │   └── fusion_model.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── benchmarker.py
├── scripts/
│   ├── build_dataset.py
│   ├── train_model.py
│   ├── run_benchmark.py
│   └── evaluate.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── reports/
├── notebooks/
└── requirements.txt
```

## Adding a New Class
```bash
mkdir -p data/raw/NewGameClass
cp ~/my_gameplay/*.mp4 data/raw/NewGameClass/
python scripts/build_dataset.py --mode all
python scripts/train_model.py --model resnet18 --num_classes <new_count> --epochs 30
python scripts/run_benchmark.py
python scripts/evaluate.py
```

## Data and Redistribution Notes
- Full datasets are not included in this repository unless redistribution is clearly permitted.
- Self-collected gameplay captures, processed training sets, and trained weights may be stored privately.
- Public documentation should describe data provenance, licensing, and intended use.

## Positioning
This project is best presented as:
- a machine learning portfolio project,
- a retro game content classification system,
- an experiment platform for visual, audio, and multimodal recognition.

That framing highlights technical skill while reducing unnecessary third-party brand exposure.
