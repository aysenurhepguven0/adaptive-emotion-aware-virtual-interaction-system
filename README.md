# Adaptive Emotion-Aware Virtual Interaction System

## Overview

This repository implements an adaptive emotion recognition system that
classifies facial expressions using deep learning. The training pipeline
is model-agnostic and supports swapping model backbones through shared
configuration. The project includes training, evaluation, single-image
inference, Grad-CAM explainability, and real-time webcam prediction.

## Implemented Models

| Model | Parameters | Input Size | Description |
|-------|-----------|------------|-------------|
| MN-Xception | ~85K | 64x64 | Lightweight custom CNN with separable convolutions |
| EfficientNet-B0 | ~4.3M | 224x224 | ImageNet pretrained with transfer learning |
| ResNet-18 | ~11.2M | 224x224 | ImageNet pretrained with partial backbone fine-tuning |
| HSEmotion | ~4.0M | 224x224 | AffectNet pretrained via timm + hsemotion (Savchenko, 2022) |

## Project Structure

```
adaptive-emotion-aware-virtual-interaction-system/
├── config.py               # Model-agnostic configuration
├── train.py                # Training pipeline (CLI)
├── evaluate.py             # Model evaluation on test set (CLI)
├── inference.py            # Single image inference (CLI)
├── webcam.py               # Real-time webcam inference (CLI)
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── __init__.py
│   └── dataset.py          # Dataset loading and transforms
│
├── models/
│   ├── __init__.py          # Model factory and checkpoint loading
│   ├── mini_xception.py     # MN-Xception architecture
│   ├── efficientnet.py      # EfficientNet-B0 architecture
│   ├── resnet.py            # ResNet-18 architecture
│   └── hsemotion_model.py   # HSEmotion (AffectNet pretrained)
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py     # Training plots and confusion matrices
│   └── grad_cam.py          # Grad-CAM explainability module
│
└── notebooks/
    ├── train_mini_xception.ipynb
    ├── train_efficientnet_B0.ipynb
    ├── train_resnet.ipynb
    ├── train_hsemotion.ipynb
    └── compare_models.ipynb    # Side-by-side model comparison
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, torchvision, OpenCV, NumPy, Pillow,
scikit-learn, matplotlib, tqdm, pandas, python-osc.

## Usage

### Training

Train a model from the command line:

```bash
python train.py \
    --model mini_xception \
    --dataset ferplus \
    --epochs 50 \
    --batch-size 32 \
    --output-dir outputs \
    --device cuda
```

Or use the Colab notebooks in `notebooks/` for GPU training.

### Model Comparison

Compare all four trained models side by side:

Open `notebooks/compare_models.ipynb` in Colab. The notebook evaluates
all models on the same FERPlus test split and produces accuracy tables,
per-class F1 charts, confusion matrices, training history overlays, and
a parameter efficiency analysis. Outputs are saved to
`results/model_comparison/`.

### Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py \
    --model mini_xception \
    --checkpoint results/mini_xception/best_mini_xception.pth \
    --dataset ferplus \
    --output-dir outputs
```

### Single Image Inference

Predict emotion from a single face image:

```bash
python inference.py \
    --model mini_xception \
    --checkpoint results/mini_xception/best_mini_xception.pth \
    --image path/to/face.jpg
```

### Grad-CAM Visualization

Generate a Grad-CAM heatmap showing which facial regions drive the
model's prediction:

```bash
python -m utils.grad_cam \
    --model mini_xception \
    --checkpoint results/mini_xception/best_mini_xception.pth \
    --image path/to/face.jpg \
    --output outputs/grad_cam.png
```

Use `--target-class <index>` to visualize a specific class instead of
the predicted one.

Supported models: `mini_xception`, `efficientnet_b0`, `resnet18`,
`hsemotion`.

### Real-Time Webcam Inference

Run live emotion prediction from your webcam:

```bash
python webcam.py \
    --model mini_xception \
    --checkpoint results/mini_xception/best_mini_xception.pth
```

Options:
- `--camera <index>` to select a specific camera device (default: 0).
- `--device cuda` to use GPU acceleration.
- Press `q` to quit the webcam window.

The webcam pipeline uses OpenCV's Haar cascade for face detection and
runs the trained model on each detected face in real time.

## Dataset

The project uses the **FERPlus** dataset with the following emotion
classes: angry, happy, neutral, sad, surprise.

The dataset is expected under `data/ferplus/` with subdirectories
for each emotion class.
