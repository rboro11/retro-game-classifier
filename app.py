"""
app.py — Gradio inference app for the Retro Game Classifier

Loads the exported EfficientNet-B0 bundle and classifies a screenshot
as either Super Mario Bros. (SMB1) or Super Mario Bros. 3 (SMB3).

Run locally:
    python app.py

For Hugging Face Spaces, this file must be in the repo root.
The export bundle must be at:  exports/EfficientNet-B0_export.pt
"""

import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── Load bundle ──────────────────────────────────────────────────────────────

BUNDLE_PATH = ROOT / "exports" / "EfficientNet-B0_export.pt"

if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Export bundle not found at {BUNDLE_PATH}.\n"
        "Run: python scripts/export_model.py"
    )

bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)

arch      = bundle["architecture"]
inf_cfg   = bundle["inference"]
training  = bundle["training"]

CLASS_NAMES = inf_cfg["class_names"]   # ['SMB1', 'SMB3']
IMG_SIZE    = inf_cfg["img_size"]       # 224
MEAN        = inf_cfg["mean"]
STD         = inf_cfg["std"]

# Rebuild model from registry
from models.transfer_models import build_model, MODEL_REGISTRY
from models.cnn_custom import MarioCNNSmall, MarioCNNMedium

key = arch["registry_key"]
n   = arch["num_classes"]

if key == "cnn_small":
    model = MarioCNNSmall(num_classes=n)
elif key == "cnn_medium":
    model = MarioCNNMedium(num_classes=n)
else:
    model, _ = build_model(key, num_classes=n)

model.load_state_dict(bundle["model_state"])
model.eval()

# ── Transform ────────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Image preprocessing ───────────────────────────────────────────────────────

def prepare_image(image: Image.Image) -> Image.Image:
    """Normalize camera/webcam images before inference.

    Handles:
    - EXIF auto-rotation (fixes upside-down or sideways phone photos)
    - Converts to RGB (strips alpha channels)
    - Center-crops to square to match training data aspect ratio
    """
    # Fix EXIF orientation (phone camera photos are often rotated)
    image = ImageOps.exif_transpose(image)

    # Ensure RGB
    image = image.convert("RGB")

    # Center-crop to square so landscape/portrait photos resize cleanly
    w, h = image.size
    crop_size = min(w, h)
    left   = (w - crop_size) // 2
    top    = (h - crop_size) // 2
    right  = left + crop_size
    bottom = top  + crop_size
    image  = image.crop((left, top, right, bottom))

    return image

# ── Inference function ───────────────────────────────────────────────────────

def classify(image: Image.Image) -> dict:
    if image is None:
        return {c: 0.0 for c in CLASS_NAMES}

    image  = prepare_image(image)
    tensor = preprocess(image).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ── Example images ───────────────────────────────────────────────────────────

examples_dir = ROOT / "examples"
EXAMPLES = [str(p) for p in sorted(examples_dir.glob("*.png"))] if examples_dir.exists() else []

# ── UI ───────────────────────────────────────────────────────────────────────

MODEL_INFO = (
    f"**Model:** {arch['display_name']} &nbsp;|&nbsp; "
    f"**Val acc:** {training['best_val_acc']:.2%} &nbsp;|&nbsp; "
    f"**Test acc:** {training['test_acc']:.2%} &nbsp;|&nbsp; "
    f"**Classes:** {', '.join(CLASS_NAMES)}"
)

DESCRIPTION = """
Upload a screenshot or gameplay frame and the model will identify
whether it's from **Super Mario Bros.** (NES, 1985) or
**Super Mario Bros. 3** (NES, 1988).

This is Phase 1 of a broader retro game classifier — trained on
self-collected gameplay footage, evaluated on held-out video clips.

> **Tip:** Direct screenshots give the best results. If using the camera,
> point it straight at the screen and crop out any borders for highest accuracy.
"""

with gr.Blocks(title="Retro Game Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎮 Retro Game Classifier")
    gr.Markdown(DESCRIPTION)
    gr.Markdown(MODEL_INFO)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload a gameplay screenshot",
                sources=["upload", "webcam", "clipboard"],
                height=300,
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Prediction",
                num_top_classes=len(CLASS_NAMES),
            )

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=image_input,
            outputs=label_output,
            fn=classify,
            cache_examples=True,
        )

    submit_btn.click(fn=classify, inputs=image_input, outputs=label_output)
    image_input.change(fn=classify, inputs=image_input, outputs=label_output)

    gr.Markdown(
        "---\n"
        "**Source:** [retro-game-classifier](https://github.com/rboro11/retro-game-classifier) · "
        "Phase 1 binary classifier · EfficientNet-B0 fine-tuned on self-collected NES gameplay footage."
    )

if __name__ == "__main__":
    demo.launch(share=False)
