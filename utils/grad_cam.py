"""
utils/grad_cam.py - Grad-CAM Visualization
============================================
Generates class activation maps showing which image regions
drive the model's emotion prediction.

Supports:
    - MiniXception (target layer: conv_final)
    - EfficientNet-B0 (target layer: backbone.features[-1])

Reference:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization.

Usage:
    python -m utils.grad_cam \
        --model mini_xception \
        --checkpoint path/to/model.pth \
        --image path/to/face.jpg \
        --output outputs/grad_cam.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.

    Hooks into a target convolutional layer to capture forward
    activations and backward gradients, then computes a weighted
    class activation map.

    Args:
        model: Trained PyTorch model.
        target_layer: The convolutional layer to visualize.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        """Store activations from the forward pass."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Store gradients from the backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W].
            target_class: Class index to visualize. If None, the
                predicted class is used.

        Returns:
            Heatmap as numpy array [H_cam, W_cam], values in [0, 1].
        """
        self.model.eval()

        # Forward pass (hooks capture activations)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # keep only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def release(self):
        """Remove registered hooks to free resources."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Resolve the target convolutional layer for Grad-CAM.

    Each model architecture has a standard layer that produces
    the most informative activation maps (typically the last
    convolutional block before global average pooling).

    Args:
        model: The model instance.
        model_name: Architecture name.

    Returns:
        The target layer module.

    Raises:
        ValueError: If the model is not supported.
    """
    name = model_name.lower()

    if name in {"mini_xception", "mn_xception"}:
        return model.conv_final
    elif name in {"efficientnet_b0", "efficientnet"}:
        return model.backbone.features[-1]
    elif name in {"resnet18", "resnet-18", "resnet"}:
        return model.backbone.layer4
    elif name in {"hsemotion", "hs_emotion", "hs-emotion"}:
        return model.backbone.blocks[-1]
    else:
        raise ValueError(
            f"No default Grad-CAM target layer for '{model_name}'. "
            "Supported models: mini_xception, efficientnet_b0, "
            "resnet18, hsemotion."
        )


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


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    emotion_color: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image as BGR numpy array [H, W, 3].
        heatmap: Grad-CAM heatmap [H_cam, W_cam], values in [0, 1].
        alpha: Blending weight for the heatmap (0 = image only,
            1 = heatmap only).
        colormap: OpenCV colormap constant for the heatmap.

    Returns:
        Blended BGR image as numpy array [H, W, 3].
    """
    heatmap_resized = cv2.resize(
        heatmap, (image.shape[1], image.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    if emotion_color is not None:
        tint = np.full_like(heatmap_colored, emotion_color)
        heatmap_colored = cv2.addWeighted(
            heatmap_colored, 0.6, tint, 0.4, 0
        )

    blended = cv2.addWeighted(
        image, 1 - alpha, heatmap_colored, alpha, 0
    )
    return blended


def generate_grad_cam(
    model_name: str,
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[str, float, np.ndarray]:
    """
    End-to-end Grad-CAM generation for a single image.

    Loads the model from a checkpoint, preprocesses the image,
    generates the Grad-CAM heatmap, and saves the overlay.

    Args:
        model_name: Architecture name (mini_xception or
            efficientnet_b0).
        checkpoint_path: Path to the model checkpoint.
        image_path: Path to the input image.
        output_path: Path to save the Grad-CAM overlay.
        target_class: Class to visualize (None = predicted class).
        device: Computation device.

    Returns:
        Tuple of (predicted_label, confidence, raw_heatmap).
    """
    from models import (
        get_model_config,
        load_model_from_checkpoint,
    )
    from data.dataset import build_transforms

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load model
    model, class_names = load_model_from_checkpoint(
        model_name, checkpoint_path, device
    )

    # Resolve input size from model config
    model_cfg = get_model_config(model_name)
    input_size = model_cfg["input_size"]

    # Preprocess the image (grayscale=True for FERPlus-trained models)
    transform = build_transforms(input_size, grayscale=True)["eval"]
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Enable gradient tracking for the input so gradients flow
    # through frozen backbone layers during Grad-CAM computation
    input_tensor.requires_grad_(True)

    # Generate Grad-CAM heatmap
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(input_tensor, target_class)
    grad_cam.release()

    # Get prediction info
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    predicted_label = class_names[pred_idx.item()]
    confidence_val = confidence.item()

    # Create and save overlay
    image_bgr = cv2.cvtColor(
        np.array(original_image), cv2.COLOR_RGB2BGR
    )
    label_for_color = predicted_label
    if target_class is not None:
        label_for_color = class_names[target_class]
    overlay = overlay_heatmap(
        image_bgr, heatmap, emotion_color=get_emotion_color(label_for_color)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    print(
        f"Grad-CAM saved: {output_path} "
        f"(prediction: {predicted_label}, "
        f"confidence: {confidence_val:.2%})"
    )

    return predicted_label, confidence_val, heatmap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Grad-CAM generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM visualization for a trained "
            "emotion recognition model."
        )
    )
    parser.add_argument(
        "--model", required=True,
        help="Model architecture (mini_xception or efficientnet_b0).",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input face image.",
    )
    parser.add_argument(
        "--output", default="outputs/grad_cam.png",
        help="Path to save the Grad-CAM overlay (default: "
             "outputs/grad_cam.png).",
    )
    parser.add_argument(
        "--target-class", type=int, default=None,
        help="Class index to visualize (default: predicted class).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for computation (default: auto-detect).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Grad-CAM generation."""
    args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    generate_grad_cam(
        model_name=args.model,
        checkpoint_path=Path(args.checkpoint),
        image_path=Path(args.image),
        output_path=Path(args.output),
        target_class=args.target_class,
        device=device,
    )


if __name__ == "__main__":
    main()
