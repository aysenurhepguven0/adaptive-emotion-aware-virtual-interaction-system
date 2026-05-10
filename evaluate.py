from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from sklearn.metrics import classification_report

from data import create_dataloaders
from models import load_model_from_checkpoint, get_model_config
from utils import plot_confusion_matrix


def evaluate(
    model_name: str,
    checkpoint: Path,
    dataset_name: str | None,
    data_root: str | None,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    model_cfg = get_model_config(model_name)
    loaders, class_names, resolved_dataset = create_dataloaders(
        dataset_name=dataset_name, data_root=data_root,
        input_size=model_cfg["input_size"],
    )
    model, class_names = load_model_from_checkpoint(
        model_name, checkpoint, device, class_names=class_names
    )
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
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print(f"\n=== Evaluation: {model_name} ({resolved_dataset}) ===")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-class F1 scores:")
    for cls in class_names:
        f1 = report[cls]["f1-score"]
        prec = report[cls]["precision"]
        rec = report[cls]["recall"]
        print(f"  {cls:<12} precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"\nMacro F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        output_path=output_dir / "confusion_matrix.png",
        normalize=True,
        title=f"{model_name} ({resolved_dataset})",
    )
    print(f"\nConfusion matrix saved → {output_dir / 'confusion_matrix.png'}")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class_f1": {cls: report[cls]["f1-score"] for cls in class_names},
    }
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
    _default_device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    parser.add_argument("--device", default=_default_device)
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
