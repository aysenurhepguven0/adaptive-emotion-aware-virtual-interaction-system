from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from .mini_xception import MiniXception, get_model as get_mini_xception
from .efficientnet import EfficientNetB0

from config import MODEL_CONFIGS

__all__ = [
    "MiniXception",
    "EfficientNetB0",
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


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> Tuple[nn.Module, Tuple[str, ...]]:
    """
    Load a trained model from a checkpoint file.

    Handles architecture-specific details such as detecting
    whether EfficientNet was trained with a channel adapter
    (in_channels=1) or direct RGB input (in_channels=3).

    Args:
        model_name: Architecture name (mini_xception,
            efficientnet_b0).
        checkpoint_path: Path to the .pth or .pt checkpoint.
        device: Target device (default: auto-detect).

    Returns:
        Tuple of (model in eval mode, class_names tuple).
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device
    )
    state_dict = checkpoint["model_state"]
    class_names = tuple(checkpoint["classes"])

    # Auto-detect in_channels for EfficientNet by checking
    # whether the checkpoint contains channel_adapter weights
    kwargs = {}
    name = model_name.lower()
    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
        has_adapter = any(
            k.startswith("channel_adapter")
            for k in state_dict
        )
        kwargs["in_channels"] = 1 if has_adapter else 3
        kwargs["freeze_backbone"] = False

    model = get_model(
        model_name, num_classes=len(class_names), **kwargs
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names
