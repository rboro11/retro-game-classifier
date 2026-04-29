[README_.md](https://github.com/user-attachments/files/27209533/README_.md)
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
3. Train a model, for example: `python scripts/train_model.py --model resnet18 --num_classes 3 --epochs 30`
4. Benchmark models with `python scripts/run_benchmark.py`

Or open the project notebook directly in Colab.

## Project Phases

| Phase | Task | Models | Data Needed |
|---|---|---|---|
| 1 | Binary visual classifier | Custom CNN | One public retro gameplay dataset + one self-collected class |
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
│   └── run_benchmark.py
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
