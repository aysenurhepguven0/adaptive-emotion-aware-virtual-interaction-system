from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from .mini_xception import MiniXception, get_model as get_mini_xception
from .efficientnet import EfficientNetB0
from .resnet import ResNet18
from .hsemotion_model import HSEmotion

from config import MODEL_CONFIGS

__all__ = [
    "MiniXception",
    "EfficientNetB0",
    "ResNet18",
    "HSEmotion",
    "get_model",
    "get_model_config",
    "load_model_from_checkpoint",
]

# Maps accepted model name variants to their config key
_MODEL_NAME_MAP = {
    "mini_xception": "mn_xception",
    "mn_xception": "mn_xception",
    "mini-xception": "mn_xception",
    "mn-xception": "mn_xception",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet": "efficientnet_b0",
    "resnet18": "resnet18",
    "resnet-18": "resnet18",
    "resnet": "resnet18",
    "hsemotion": "hsemotion",
    "hs_emotion": "hsemotion",
    "hs-emotion": "hsemotion",
}


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Create a model instance by architecture name.

    Args:
        model_name: Architecture identifier (e.g. 'mini_xception',
            'efficientnet_b0').
        num_classes: Number of output classes.
        **kwargs: Additional model-specific parameters.

    Returns:
        Initialized model (untrained).

    Raises:
        ValueError: If the model name is not recognized.
    """
    name = model_name.lower()

    if name in {"mini_xception", "mn_xception",
                "mini-xception", "mn-xception"}:
        return get_mini_xception(num_classes=num_classes, **kwargs)

    elif name in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
        return EfficientNetB0(
            num_classes=num_classes,
            in_channels=kwargs.get("in_channels", 3),
            freeze_backbone=kwargs.get("freeze_backbone", False),
            unfreeze_last_n=kwargs.get("unfreeze_last_n", 2),
        )

    elif name in {"resnet18", "resnet-18", "resnet"}:
        return ResNet18(
            num_classes=num_classes,
            in_channels=kwargs.get("in_channels", 3),
            freeze_backbone=kwargs.get("freeze_backbone", False),
            unfreeze_last_n=kwargs.get("unfreeze_last_n", 2),
        )

    elif name in {"hsemotion", "hs_emotion", "hs-emotion"}:
        return HSEmotion(
            num_classes=num_classes,
            in_channels=kwargs.get("in_channels", 3),
            freeze_backbone=kwargs.get("freeze_backbone", False),
            unfreeze_last_n=kwargs.get("unfreeze_last_n", 2),
            affectnet_pretrained=kwargs.get(
                "affectnet_pretrained", True
            ),
        )

    raise ValueError(f"Unknown model name: {model_name}")


def get_model_config(model_name: str) -> dict:
    """
    Get the configuration dict for a model architecture.

    Looks up the model's entry in config.MODEL_CONFIGS using
    a normalized name mapping.

    Args:
        model_name: Architecture identifier.

    Returns:
        Configuration dict with keys like 'input_size',
        'display_name', 'dropout', etc.

    Raises:
        ValueError: If no config is found for the model name.
    """
    config_key = _MODEL_NAME_MAP.get(model_name.lower())
    if config_key is None:
        raise ValueError(
            f"No config found for model: {model_name}. "
            f"Known models: {list(_MODEL_NAME_MAP.keys())}"
        )
    return MODEL_CONFIGS[config_key]


def _infer_num_classes(state_dict: dict, model_name: str) -> int:
    """Infer the number of output classes from a raw state dict."""
    name = model_name.lower()
    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
        key = "backbone.classifier.4.weight"
    elif name in {"mini_xception", "mn_xception",
                  "mini-xception", "mn-xception"}:
        key = "fc.weight"
    elif name in {"resnet18", "resnet-18", "resnet"}:
        key = "backbone.fc.4.weight"
    elif name in {"hsemotion", "hs_emotion", "hs-emotion"}:
        key = "backbone.classifier.4.weight"
    else:
        raise ValueError(
            f"Cannot infer num_classes for model: {model_name}"
        )
    if key not in state_dict:
        raise KeyError(
            f"Expected key '{key}' not found in state dict"
        )
    return state_dict[key].shape[0]


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    class_names: Tuple[str, ...] | None = None,
) -> Tuple[nn.Module, Tuple[str, ...]]:
    """
    Load a trained model from a checkpoint file.

    Supports two checkpoint formats:
        1. Wrapped dict with 'model_state' and 'classes' keys
           (produced by train.py).
        2. Raw state dict (produced by notebook training).

    Handles architecture-specific details such as detecting
    whether EfficientNet was trained with a channel adapter
    (in_channels=1) or direct RGB input (in_channels=3).

    Args:
        model_name: Architecture name (mini_xception,
            efficientnet_b0).
        checkpoint_path: Path to the .pth or .pt checkpoint.
        device: Target device (default: auto-detect).
        class_names: Optional class names override. Required
            when loading a raw state dict checkpoint that does
            not contain class information.

    Returns:
        Tuple of (model in eval mode, class_names tuple).
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Detect checkpoint format
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        # Wrapped format from train.py
        state_dict = checkpoint["model_state"]
        if class_names is None:
            class_names = tuple(checkpoint["classes"])
    else:
        # Raw state dict (e.g. from notebook training)
        state_dict = checkpoint
        if class_names is None:
            num_classes = _infer_num_classes(state_dict, model_name)
            class_names = tuple(
                f"class_{i}" for i in range(num_classes)
            )

    # Auto-detect in_channels for transfer learning models by
    # checking whether the checkpoint contains channel_adapter weights
    kwargs = {}
    name = model_name.lower()
    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnet",
                "resnet18", "resnet-18", "resnet",
                "hsemotion", "hs_emotion", "hs-emotion"}:
        has_adapter = any(
            k.startswith("channel_adapter")
            for k in state_dict
        )
        kwargs["in_channels"] = 1 if has_adapter else 3
        kwargs["freeze_backbone"] = False
        # Skip AffectNet download when loading from checkpoint
        if name in {"hsemotion", "hs_emotion", "hs-emotion"}:
            kwargs["affectnet_pretrained"] = False

    model = get_model(
        model_name, num_classes=len(class_names), **kwargs
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names
