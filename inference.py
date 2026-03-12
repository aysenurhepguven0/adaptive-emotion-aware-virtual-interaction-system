from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn

from data.dataset import build_transforms, resolve_dataset


def get_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        from models import get_model as model_factory
    except ImportError as exc:
        raise ImportError("Model definitions are not available yet.") from exc
    return model_factory(model_name, num_classes=num_classes)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Tuple[str, ...]]:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    return checkpoint_data["model_state"], tuple(checkpoint_data["classes"])


def predict_image(
    model_name: str,
    checkpoint_path: Path,
    image_path: Path,
    dataset_name: str | None,
    device: torch.device,
) -> Tuple[str, float]:
    dataset_name, _, metadata = resolve_dataset(dataset_name)
    model_state, class_names = load_checkpoint(checkpoint_path, device)

    model = get_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(model_state)
    model.eval()

    transform = build_transforms(metadata["input_size"], metadata["grayscale"])["eval"]
    image = Image.open(image_path).convert("RGB")
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
