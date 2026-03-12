from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_history(
    history: Mapping[str, Iterable[float]],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="Train")
    axes[0].plot(epochs, history.get("val_loss", []), label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc", []), label="Train")
    axes[1].plot(epochs, history.get("val_acc", []), label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Iterable[str],
    output_path: str | Path | None = None,
    normalize: bool = True,
    title: str | None = None,
) -> plt.Figure:
    labels = list(range(len(list(class_names))))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig
