# 🏗️ Illegal Construction Detector

> **AI-powered satellite/aerial image analysis for detecting buildings in restricted zones using semantic segmentation.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-See%20LICENSE-green)](#license)

---

## 📌 Overview

This project uses a deep learning segmentation model to automatically identify buildings in aerial or satellite imagery and flag those that fall within user-defined restricted/zoning areas. It is designed for urban planning, land administration, and regulatory compliance workflows.

**Key capabilities:**
- Semantic segmentation of buildings from high-resolution aerial images
- Patch-based inference for large images of arbitrary size
- Zoning mask overlay to classify buildings as **legal** or **illegal**
- REST API backend (Flask) with base64 image response for easy frontend integration
- Visual overlay output highlighting illegal structures in red and legal structures in green

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Architecture | U-Net |
| Encoder | EfficientNet-B3 (ImageNet pretrained) |
| Loss Function | Dice Loss (binary) |
| Optimizer | Adam (lr = 1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Input | RGB patches of 256 × 256 px |
| Output | Binary segmentation mask (building / non-building) |

The model was trained on the [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) for 30 epochs with augmentations including horizontal/vertical flips, random 90° rotation, and brightness/contrast jitter.

---

## 🗂️ Repository Structure

```
├── app.py                  # Flask API server
├── best_model.pth          # Trained model weights
├── requirements.txt        # Python dependencies
├── sourcecode.ipynb        # Model training notebook (Kaggle)
├── inference_script.ipynb  # Standalone inference & visualization notebook
├── static/
│   └── index.html          # Frontend UI (served by Flask)
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU *(optional but recommended)*

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/illegal-construction-detector.git
cd illegal-construction-detector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the model weights in the project root
#    Ensure best_model.pth is present (download from Releases or train your own)
```

---

## 🚀 Usage

### Option 1 — Flask Web API

```bash
python app.py
```

The server starts at `http://localhost:5000`. Navigate there in your browser to use the UI, or send requests directly:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/aerial_image.png"
```

**Response JSON:**

```json
{
  "verdict": "ILLEGAL CONSTRUCTION DETECTED",
  "illegal_count": 3,
  "legal_count": 12,
  "total_count": 15,
  "illegal_percent": 8.42,
  "device": "cuda",
  "original": "<base64_png>",
  "mask": "<base64_png>",
  "overlay": "<base64_png>"
}
```

### Option 2 — Jupyter Notebook Inference

Open `inference_script.ipynb` and update the following variables:

```python
MODEL_PATH = "best_model.pth"    # path to weights
IMAGE_PATH = "test_image.png"    # path to your aerial image
```

Run all cells to generate the building mask, zoning overlay, and illegal building highlights.

---

## 🗺️ Zoning Configuration

By default, the right half of any image is treated as a restricted zone. To customise this, modify the `create_zoning_mask` function in `app.py` or the inference notebook:

```python
def create_zoning_mask(shape):
    h, w = shape
    zoning = np.zeros((h, w), dtype=np.uint8)
    # Example: mark a rectangular region as restricted
    zoning[100:400, 200:600] = 1
    return zoning
```

Any building whose pixels overlap with the restricted zone (`zoning == 1`) is classified as **illegal**.

---

## 🏋️ Training

The full training pipeline is available in `sourcecode.ipynb`, designed to run on Kaggle with GPU acceleration.

**Dataset:** [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)

**Training configuration:**

```python
PATCH_SIZE  = 256
BATCH_SIZE  = 8
EPOCHS      = 30
LEARNING_RATE = 1e-4
```

**Augmentations:** Horizontal flip, vertical flip, random 90° rotation, brightness/contrast jitter

**Metrics tracked:** Dice Loss · IoU Score

To retrain, update `BASE_PATH` in the notebook to your dataset location and run all cells. The best-performing checkpoint (highest IoU) is saved as `best_model.pth`.

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation IoU | Tracked per epoch |
| Inference mode | Patch-based (256×256), supports arbitrary image sizes |
| Supported formats | PNG, JPG, TIFF |

---

## 🔧 Dependencies

```
flask
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
pillow
numpy
```

Install all with:
```bash
pip install -r requirements.txt
```

> **Note for Windows users:** If you encounter a `fbgemm.dll` error when importing PyTorch, ensure you have the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes, then submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a Pull Request

---

## 📄 License

See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) by Balraj Ashwath
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii
- [Albumentations](https://albumentations.ai/) augmentation library
