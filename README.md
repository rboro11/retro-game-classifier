# Retro Game Classifier

A computer vision classifier for retro game screenshots, currently trained on **Super Mario Bros. 1, Super Mario Bros. 2, and Super Mario Bros. 3** with a Gradio deployment workflow.[file:490]

## Overview

This repository contains a local-runtime training and deployment pipeline for classifying gameplay frames from classic NES Mario titles.[file:490] The current notebook workflow builds processed frame datasets, creates balanced train/validation/test splits, trains an EfficientNet-B0 model, evaluates it, exports a deployment bundle, and prepares assets for a Hugging Face Space.[file:490]

## Current build

The latest build is a **3-class** image classifier with labels `SMB1`, `SMB2`, and `SMB3`.[file:490] The exported deployment bundle records the architecture as `efficientnetb0`, `numclasses` as 3, class names `SMB1`, `SMB2`, `SMB3`, and image size 224.[file:490]

## Dataset and splits

The current dataset uses gameplay frames from **SMB1, SMB2, and SMB3**.[file:490] SMB1 is split by gameplay session, SMB3 is split by source video clip, and SMB2 currently comes from a single **full-playthrough** source video that provides broad gameplay coverage across the game; because only one SMB2 clip was available in this build, its train/validation/test sets were created with a chronological frame-level split rather than across multiple independent clips.[file:490]

After capping to 1,000 examples per class in the split builder, the resulting balanced dataset contains **2,132 training samples, 455 validation samples, and 408 test samples** across the three classes.[file:490] The split logs report near-even class totals: SMB1 999, SMB2 998, and SMB3 998.[file:490]

## Results

The current best model is **EfficientNet-B0** fine-tuned for 3-way classification.[file:490] Training stopped early at epoch 8, with best validation accuracy **0.9956** and reported training time of about **4.7 minutes** on the recorded run.[file:490]

On the held-out test set, the notebook reports **test accuracy 1.0000**, **macro F1 1.0000**, **macro precision 1.0000**, and **macro recall 1.0000**.[file:490] The per-class report shows precision, recall, and F1 of **1.00** for SMB1, SMB2, and SMB3 on supports of 148, 149, and 111 samples respectively.[file:490]

## Repository structure

The notebook references a project structure including `app.py`, `configs/config.yaml`, `scripts/`, `src/`, `exports/`, `docs/`, and `notebooks/`.[file:490] The scripts used in the workflow include dataset building, evaluation, benchmarking, frame extraction, and export utilities such as `builddataset.py`, `evaluate.py`, `runbenchmark.py`, `splicevideo.py`, and `chunksplicevideo.py`.[file:490]

## Quick start

1. Clone the repository.
2. Install dependencies from `requirements.txt`.[file:490]
3. Place raw class data under `data/raw/SMB1`, `data/raw/SMB2`, and `data/raw/SMB3`.[file:490]
4. Build processed frames and splits with the dataset scripts.[file:490]
5. Train the model, evaluate it, and export the best bundle for deployment.[file:490]

Example commands:

```bash
git clone https://github.com/rboro11/retro-game-classifier.git
cd retro-game-classifier
pip install -r requirements.txt
python scripts/builddataset.py --mode frames --fps 0.3
python scripts/builddataset.py --mode splits --max_per_class 1000
python scripts/trainmodel.py --model efficientnetb0 --numclasses 3 --epochs 20 --batchsize 32 --lr 3e-4
python scripts/evaluate.py
python scripts/exportmodel.py
```

## Deployment

The notebook workflow exports `exports/EfficientNet-B0export.pt` together with a metadata sidecar and then prepares deployment assets for a Hugging Face Space.[file:490] It also uses `classes.csv` alongside the exported model bundle during deployment preparation, which is helpful when class metadata needs an explicit source of truth.[file:490]

The checked-in app file still appears to contain older binary-project wording in its description text, including references to SMB1 vs SMB3 and “Phase 1.”[file:474] For consistency with the current notebook and exported bundle, the app description should be updated to reflect the current **SMB1/SMB2/SMB3** classifier.[file:474][file:490]

## Notes

The current SMB2 data is broad from a gameplay-content perspective because it comes from a full playthrough, but its evaluation split is still derived from one recording rather than multiple independent source clips.[file:490] That is best described as broad gameplay coverage with limited source-recording diversity in the present build.[file:490]
