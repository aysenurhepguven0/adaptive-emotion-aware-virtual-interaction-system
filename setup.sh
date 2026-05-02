#!/bin/bash
# Setup script: create venv and install all dependencies
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Upgrading pip..."
.venv/bin/pip install --upgrade pip

echo "Installing PyTorch (with MPS support for Apple Silicon)..."
.venv/bin/pip install torch torchvision

echo "Installing remaining dependencies..."
.venv/bin/pip install timm numpy pandas scikit-learn matplotlib opencv-python Pillow tqdm python-osc

echo ""
echo "Setup complete! To launch the application:"
echo "  bash run.sh"
echo ""
echo "For webcam inference directly:"
echo "  .venv/bin/python webcam.py --model mini_xception --checkpoint <path_to_checkpoint>"
