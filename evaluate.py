from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from data import create_dataloaders
from utils import plot_confusion_matrix


def get_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        from models import get_model as model_factory
    except ImportError as exc:
        raise ImportError("Model definitions are not available yet.") from exc
    return model_factory(model_name, num_classes=num_classes)


def evaluate(
    model_name: str,
    checkpoint: Path,
    dataset_name: str | None,
    data_root: str | None,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    loaders, class_names, resolved_dataset = create_dataloaders(
        dataset_name=dataset_name, data_root=data_root
    )
    model = get_model(model_name, num_classes=len(class_names)).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()

    y_true = []
    y_pred = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / max(total_samples, 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        output_path=output_dir / "confusion_matrix.png",
        normalize=True,
        title=f"{model_name} ({resolved_dataset})",
    )

    metrics = {"accuracy": accuracy}
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emotion recognition models.")
    parser.add_argument("--model", required=True, help="Model name (e.g. mini_xception).")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--dataset", default=None, help="Dataset name override.")
    parser.add_argument("--data-root", default=None, help="Override dataset root path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save artifacts.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_name=args.model,
        checkpoint=Path(args.checkpoint),
        dataset_name=args.dataset,
        data_root=args.data_root,
        device=torch.device(args.device),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
