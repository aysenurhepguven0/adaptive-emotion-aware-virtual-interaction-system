from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TRAINING_CONFIG
from data import create_dataloaders
from utils import plot_training_history


def get_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        from models import get_model as model_factory
    except ImportError as exc:
        raise ImportError("Model definitions are not available yet.") from exc
    return model_factory(model_name, num_classes=num_classes)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def train_model(
    model_name: str,
    dataset_name: str | None,
    data_root: str | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, list]:
    loaders, class_names, resolved_dataset = create_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
    )

    model = get_model(model_name, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(epochs):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_{resolved_dataset}.pt"
    torch.save({"model_state": model.state_dict(), "classes": class_names}, checkpoint_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    plot_training_history(history, output_dir / "training_history.png")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train emotion recognition models.")
    parser.add_argument("--model", required=True, help="Model name (e.g. mini_xception).")
    parser.add_argument("--dataset", default=None, help="Dataset name override.")
    parser.add_argument("--data-root", default=None, help="Override dataset root path.")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--output-dir", default="outputs", help="Directory to save artifacts.")
    _default_device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    parser.add_argument("--device", default=_default_device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
