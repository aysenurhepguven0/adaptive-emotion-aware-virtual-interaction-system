from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from data.dataset import build_transforms, resolve_dataset
from models import load_model_from_checkpoint, get_model_config

EMOTION_CLASSES = ("angry", "happy", "neutral", "sad", "surprise")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
FACE_PADDING = 0.2


def _crop_face(image: Image.Image) -> Image.Image:
    """Detect the largest face and return a padded crop. Falls back to
    the full image if no face is detected."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        print("[WARNING] No face detected – using full image.")
        return image

    # pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)
    ih, iw = bgr.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    return image.crop((x1, y1, x2, y2))


def predict_image(
    model_name: str,
    checkpoint_path: Path,
    image_path: Path,
    dataset_name: str | None,
    device: torch.device,
) -> Tuple[str, float]:
    dataset_name, _, metadata = resolve_dataset(dataset_name)
    model_cfg = get_model_config(model_name)

    model, class_names = load_model_from_checkpoint(
        model_name, checkpoint_path, device, class_names=EMOTION_CLASSES
    )

    # Use the model's expected input size (e.g. 224 for ResNet-18),
    # not the dataset's native size (48 for FERPlus) — they differ.
    transform = build_transforms(model_cfg["input_size"], metadata["grayscale"])["eval"]
    image = Image.open(image_path).convert("RGB")
    image = _crop_face(image)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, index = probs.max(dim=1)

    return class_names[index.item()], confidence.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--model", required=True, help="Model name (e.g. mini_xception).")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--dataset", default=None, help="Dataset name override.")
    _default_device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    parser.add_argument("--device", default=_default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, confidence = predict_image(
        model_name=args.model,
        checkpoint_path=Path(args.checkpoint),
        image_path=Path(args.image),
        dataset_name=args.dataset,
        device=torch.device(args.device),
    )
    print(f"Prediction: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
