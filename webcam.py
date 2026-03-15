"""
webcam.py - Real-Time Webcam Emotion Recognition
==================================================
Captures video from a webcam, detects faces using OpenCV's Haar
cascade, and predicts emotions in real time using a trained model.

Supports:
    - MiniXception (64x64 input)
    - EfficientNet-B0 (224x224 input)

Usage:
    python webcam.py \
        --model mini_xception \
        --checkpoint results/mini_xception/best_mini_xception.pth \
        --classes "angry,happy,neutral,sad,suprise"

Controls:
    q - quit the application
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from data.dataset import build_transforms
from models import get_model_config, load_model_from_checkpoint

# Padding ratio around detected face bounding boxes. A value of
# 0.2 adds 20% of the face width/height on each side to capture
# surrounding context, which improves recognition accuracy.
FACE_PADDING_RATIO = 0.2

# Minimum face size in pixels for the Haar cascade detector.
# Faces smaller than this are ignored to reduce false positives.
MIN_FACE_SIZE = (48, 48)

# Haar cascade detector parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
FPS_COLOR = (255, 255, 0)  # cyan FPS counter

EMOTION_COLORS = {
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 0, 255),
    "suprise": (255, 0, 255),
    "fear": (0, 165, 255),
    "disgust": (0, 255, 0),
    "neutral": (255, 255, 255),
}


def get_emotion_color(label: str) -> Tuple[int, int, int]:
    """Resolve the BGR color for a predicted emotion label."""
    return EMOTION_COLORS.get(label.lower(), (0, 255, 0))


def build_face_detector() -> cv2.CascadeClassifier:
    """
    Initialize OpenCV's Haar cascade face detector.

    Returns:
        Configured CascadeClassifier instance.

    Raises:
        RuntimeError: If the cascade file cannot be loaded.
    """
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from: {cascade_path}"
        )

    return detector


def detect_faces(
    detector: cv2.CascadeClassifier,
    frame: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a video frame.

    Args:
        detector: Haar cascade face detector.
        frame: BGR video frame from OpenCV.

    Returns:
        List of (x, y, w, h) bounding boxes for detected faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )

    # Convert to list of tuples (detectMultiScale returns ndarray)
    if len(faces) == 0:
        return []
    return [tuple(face) for face in faces]


def crop_face_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = FACE_PADDING_RATIO,
) -> np.ndarray:
    """
    Crop a face region from the frame with extra padding.

    Adds padding around the bounding box to include context
    (forehead, chin, etc.) which helps recognition accuracy.
    The crop is clipped to frame boundaries.

    Args:
        frame: BGR video frame.
        bbox: Face bounding box (x, y, w, h).
        padding: Fraction of face size to add as padding.

    Returns:
        Cropped BGR face image.
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    # Compute padded bounding box
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)

    return frame[y1:y2, x1:x2]


def preprocess_face(
    face_bgr: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess a cropped face for model inference.

    Converts the BGR face crop to a PIL RGB image and applies
    the evaluation transform pipeline (resize, grayscale
    conversion, normalization).

    Args:
        face_bgr: Cropped face image in BGR format.
        transform: torchvision evaluation transform.
        device: Target device for the tensor.

    Returns:
        Preprocessed tensor of shape [1, C, H, W].
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tensor = transform(face_pil).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def predict_emotion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names: Tuple[str, ...],
) -> Tuple[str, float]:
    """
    Run emotion prediction on a preprocessed face tensor.

    Args:
        model: Trained model in eval mode.
        input_tensor: Preprocessed face tensor [1, C, H, W].
        class_names: Tuple of emotion class labels.

    Returns:
        Tuple of (predicted_emotion, confidence).
    """
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, index = probs.max(dim=1)
    return class_names[index.item()], confidence.item()


def draw_prediction(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    """
    Draw a bounding box and prediction label on the frame.

    Args:
        frame: BGR video frame (modified in place).
        bbox: Face bounding box (x, y, w, h).
        label: Predicted emotion label.
        confidence: Prediction confidence (0-1).
    """
    x, y, w, h = bbox
    color = get_emotion_color(label)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label} ({confidence:.0%})"
    text_y = y - 10 if y - 10 > 20 else y + h + 25
    cv2.putText(
        frame, text, (x, text_y),
        FONT, FONT_SCALE, color, FONT_THICKNESS,
    )


def run_webcam(
    model_name: str,
    checkpoint_path: Path,
    camera_index: int = 0,
    device: torch.device | None = None,
    class_names: tuple[str, ...] | None = None,
) -> None:
    """
    Main webcam inference loop.

    Opens the webcam, detects faces in each frame, runs emotion
    prediction, and displays results in a window. Press 'q' to
    exit.

    Args:
        model_name: Model architecture name.
        checkpoint_path: Path to the trained model checkpoint.
        camera_index: OpenCV camera device index.
        device: Computation device (default: auto-detect).
        class_names: Optional class label override.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load model and preprocessing pipeline
    print(f"Loading model: {model_name}")
    model, class_names = load_model_from_checkpoint(
        model_name, checkpoint_path, device,
        class_names=class_names,
    )
    print(f"Classes: {class_names}")

    model_cfg = get_model_config(model_name)
    input_size = model_cfg["input_size"]
    transform = build_transforms(input_size, grayscale=True)["eval"]

    # Initialize face detector
    face_detector = build_face_detector()

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera (index={camera_index}).")
        return

    print("Webcam started. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)

        # Detect faces in the current frame
        faces = detect_faces(face_detector, frame)

        # Run inference on each detected face
        for bbox in faces:
            face_crop = crop_face_with_padding(frame, bbox)

            # Skip very small crops that would not produce
            # meaningful predictions
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            input_tensor = preprocess_face(
                face_crop, transform, device
            )
            label, confidence = predict_emotion(
                model, input_tensor, class_names
            )
            draw_prediction(frame, bbox, label, confidence)

        # Compute and display FPS
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            FONT, FONT_SCALE, FPS_COLOR, FONT_THICKNESS,
        )

        cv2.imshow("Emotion Recognition", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for webcam inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Real-time emotion recognition from webcam using "
            "a trained model."
        )
    )
    parser.add_argument(
        "--model", required=True,
        help="Model architecture (mini_xception or "
             "efficientnet_b0).",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for computation (default: auto-detect).",
    )
    parser.add_argument(
        "--classes", default=None,
        help="Comma-separated class names (e.g. "
             "'angry,happy,neutral,sad,surprise'). Required for "
             "raw state-dict checkpoints without embedded class "
             "info.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for webcam emotion recognition."""
    args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    class_names = None
    if args.classes:
        class_names = tuple(
            c.strip() for c in args.classes.split(",")
        )

    run_webcam(
        model_name=args.model,
        checkpoint_path=Path(args.checkpoint),
        camera_index=args.camera,
        device=device,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
