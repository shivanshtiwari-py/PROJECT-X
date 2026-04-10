import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import io
import cv2
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder="static")

# ── Model — exact same as training ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights=None,  # no pretrained needed at inference
    in_channels=3,
    classes=1,
    activation=None,  # raw logits, same as training
)

MODEL_PATH = "best_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weight_only=True))
    print(f"✅ Model loaded — device: {device}")
else:
    print("⚠️  best_model.pth not found — running in demo mode")

model.to(device)
model.eval()

# ── Preprocessing — matches val_transform from training ───────
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def preprocess(pil_img, patch_size=256):
    """Resize to nearest multiple of patch_size, normalize, tensorize."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # Pad to multiple of patch_size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    padded = np.zeros((new_h, new_w, 3), dtype=np.float32)
    padded[:h, :w] = img

    # Normalize
    padded = padded / 255.0
    padded = (padded - mean) / std

    tensor = torch.tensor(padded).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, (h, w)


def run_inference(pil_img, patch_size=256):
    """Run patch-based inference matching training patch extraction."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # Pad to multiple of patch_size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded[:h, :w] = img

    full_mask = np.zeros((new_h, new_w), dtype=np.float32)

    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = (
                padded[i : i + patch_size, j : j + patch_size].astype(np.float32)
                / 255.0
            )
            patch = (patch - mean) / std
            tensor = (
                torch.tensor(patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                out = model(tensor)
                prob = torch.sigmoid(out).squeeze().cpu().numpy()

            full_mask[i : i + patch_size, j : j + patch_size] = prob

    return full_mask[:h, :w], img


# ── Zoning + illegal detection (from your notebook) ───────────
def create_zoning_mask(shape):
    h, w = shape
    zoning = np.zeros((h, w), dtype=np.uint8)
    zoning[:, w // 2 :] = 1
    return zoning


def detect_illegal_buildings(binary_mask, zoning_mask):
    num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    illegal, legal = [], []
    for label in range(1, num_labels):
        building_pixels = labels == label
        if (building_pixels & (zoning_mask == 1)).any():
            illegal.append(label)
        else:
            legal.append(label)
    return illegal, legal, labels


def to_base64(arr_or_img):
    if isinstance(arr_or_img, np.ndarray):
        img = Image.fromarray(arr_or_img.astype(np.uint8))
    else:
        img = arr_or_img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    pil_img = Image.open(file.stream).convert("RGB")

    # Run patch-based segmentation
    prob_mask, orig_rgb = run_inference(pil_img, patch_size=256)
    binary_mask = (prob_mask > 0.5).astype(np.uint8)

    # Zoning-based illegal detection
    zoning_mask = create_zoning_mask(binary_mask.shape)
    illegal, legal, labels = detect_illegal_buildings(binary_mask, zoning_mask)

    # Build overlay: illegal=red, legal=green
    overlay = orig_rgb.copy()
    for lbl in illegal:
        overlay[labels == lbl] = [255, 0, 0]
    for lbl in legal:
        overlay[labels == lbl] = [0, 200, 100]

    total = len(illegal) + len(legal)
    illegal_pct = round(float(binary_mask.mean() * 100), 2)
    verdict = "ILLEGAL CONSTRUCTION DETECTED" if illegal else "NO VIOLATION DETECTED"

    return jsonify(
        {
            "verdict": verdict,
            "illegal_count": len(illegal),
            "legal_count": len(legal),
            "total_count": total,
            "illegal_percent": illegal_pct,
            "device": str(device),
            "original": to_base64(orig_rgb),
            "mask": to_base64((prob_mask * 255).astype(np.uint8)),
            "overlay": to_base64(overlay),
        }
    )


if __name__ == "__main__":
    print(f"Running on: {device}")
    app.run(debug=True, port=5000)
