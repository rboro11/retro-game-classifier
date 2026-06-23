# Retro Game Classifier — Project Plan

## Goal
Build a multi-modal deep learning system that classifies which retro game a given input came from, using screenshots, audio clips, or short video clips. The architecture should be plug-and-play so that a new class can be added by creating a new data directory and rerunning the dataset build and training pipeline.

## Positioning
This project should be positioned as a research and portfolio system for gameplay-content classification. It should not be presented as an official product, affiliated product, or publisher-endorsed application. Any third-party game names, brands, characters, screenshots, audio, or related media referenced during experimentation remain the property of their respective rights holders.

## Recommended Development Strategy
Prioritize one polished visual classification pipeline first, then add carefully chosen extensions.

Recommended order:
1. Binary screenshot classifier
2. Small multi-class screenshot classifier
3. Benchmarking and reproducibility improvements
4. One advanced extension, preferably audio
5. Optional multimodal fusion
6. Video classification only after the above is stable

## Phase Roadmap

### Phase 1 — Binary Visual Classifier
**Goal:** Distinguish between two visually distinct retro platformer classes.

**Input:** Single screenshot (for example, 256x240 PNG or a normalized equivalent)

**Model:** Tiny custom CNN baseline

**Data needed:**
- One public gameplay frame dataset
- One self-collected comparison class

**Deliverable:**
- Working train/eval loop
- Reproducible splits
- Baseline accuracy and confusion matrix
- Documented preprocessing pipeline

### Phase 2 — Multi-Class Screenshot Classifier (3-5 classes)
**Goal:** Distinguish among several retro platformer classes from screenshots.

**Input:** Single screenshot

**Models:**
- Custom CNN
- ResNet-18 fine-tune
- EfficientNet-B0

**Data needed:**
- Public screenshot datasets where permitted
- Self-collected gameplay captures
- Optional screenshot APIs where licensing terms allow use

**Deliverable:**
- Model comparison table
- Per-class metrics
- Confusion matrices
- Error analysis examples

### Phase 3 — Extended Visual Classifier (10-20 classes)
**Goal:** Expand to multi-era title recognition.

**Input:** Single screenshot

**Models:**
- ResNet-50
- EfficientNet-B2 or B3
- Vision Transformer (ViT-B/16)

**Data needed:**
- Public screenshot repositories and APIs
- Self-collected captures for underrepresented classes

**Deliverable:**
- Larger-scale benchmark
- Class-balance strategy
- Data provenance documentation

### Phase 4 — Audio Classifier
**Goal:** Identify a game from a short music or sound-effect clip.

**Input:** Audio spectrogram or mel-spectrogram representation

**Models:**
- Small CNN on spectrograms
- Optional transfer model on spectrogram images

**Data needed:**
- Public or research-available retro game audio datasets
- Self-collected audio clips where usage is allowed

**Deliverable:**
- Standalone audio classifier
- Spectrogram pipeline
- Audio benchmark report

### Phase 5 — Video Classifier
**Goal:** Identify a game from a short gameplay clip.

**Input:** Frame stack or short video tensor

**Models:**
- CNN + LSTM
- 3D-ResNet
- Optional transformer-based temporal model

**Data needed:**
- Public gameplay clip datasets
- Self-collected recordings

**Deliverable:**
- Temporal model benchmark
- Comparison versus single-frame models

### Phase 6 — Late Fusion Multi-Modal
**Goal:** Combine image and audio, with optional video, into one prediction system.

**Input:** Screenshot plus audio clip, optionally short video

**Models:**
- Separate encoders
- Concatenation or attention-based fusion head

**Data needed:**
- Paired multi-modal samples across supported classes

**Deliverable:**
- Fusion benchmark
- Ablation study comparing image-only, audio-only, and fused systems

## Public vs Self-Collected Data

| Category | Source Type | Examples |
|---|---|---|
| Public | Open or research datasets | Gameplay frame sets, audio datasets, public clip datasets |
| Semi-public | APIs or datasets with access terms | Screenshot APIs, gated research datasets |
| Self-collected | Personal captures and recordings | Local gameplay captures, extracted frames, user-generated clips |

For every dataset used, document:
- source URL,
- access conditions,
- redistribution rules,
- whether derived outputs can be shared publicly.

## Data Directory Convention
```text
data/
  raw/
    ClassA/
    ClassB/
    ClassC/
    ...
  processed/
    frames/
    spectrograms/
    splits/
```

Adding a new class should require only:
1. Creating a new folder in `data/raw/<ClassName>/`
2. Dropping screenshots, clips, or recordings into that folder
3. Rebuilding the dataset
4. Retraining and benchmarking

## Recommended Capture Setup
1. Capture gameplay from a legal local source using a capture card or recording software.
2. Record varied scenes rather than one repeated level segment.
3. Sample frames conservatively to avoid near-duplicate images.
4. Target roughly 500-2000 frames per class for early image experiments.
5. Keep raw captures private unless redistribution rights are clear.

## Portfolio and Commercialization Guidance
Publicly showcase:
- architecture design,
- training code,
- evaluation methodology,
- benchmark results,
- limited demo artifacts.

Keep private when appropriate:
- large raw datasets,
- sensitive processed datasets,
- production-ready model weights,
- commercialization-specific deployment details.

## Year-One Outcome
A strong year-one outcome is:
- one polished image classifier pipeline,
- a robust benchmarking system,
- one advanced extension such as audio classification,
- a simple demo application,
- clear documentation of data provenance and licensing boundaries.
