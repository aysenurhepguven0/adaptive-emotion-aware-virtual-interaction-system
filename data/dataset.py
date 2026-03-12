from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DATA_CONFIG, TRAINING_CONFIG

DEFAULT_DATASET = "ferplus"

DATASET_METADATA = {
    "fer2013": {"subdir": "fer2013", "input_size": 48, "grayscale": True},
    "ferplus": {"subdir": "ferplus", "input_size": 48, "grayscale": True},
    "raf_db": {"subdir": "raf_db", "input_size": 224, "grayscale": False},
    "ckplus": {"subdir": "ckplus", "input_size": 224, "grayscale": False},
}


@dataclass(frozen=True)
class DatasetBundle:
    train: Subset
    val: Subset
    test: Subset
    class_names: Tuple[str, ...]
    dataset_name: str


def resolve_dataset(dataset_name: str | None = None, data_root: str | None = None) -> Tuple[str, Path, Dict]:
    name = (dataset_name or DATA_CONFIG.get("dataset_name") or DEFAULT_DATASET).lower()
    if name not in DATASET_METADATA:
        raise ValueError(f"Unsupported dataset: {name}")

    dataset_roots = DATA_CONFIG.get("dataset_roots", {})
    if name in dataset_roots:
        dataset_root = Path(dataset_roots[name])
    else:
        base_root = Path(data_root or DATA_CONFIG.get("dataset_root", "data"))
        dataset_root = base_root / DATASET_METADATA[name]["subdir"]

    return name, dataset_root, DATASET_METADATA[name]


def build_transforms(input_size: int, grayscale: bool) -> Dict[str, transforms.Compose]:
    base_transforms = [transforms.Resize((input_size, input_size))]
    if grayscale:
        base_transforms.append(transforms.Grayscale(num_output_channels=3))

    train_transforms = transforms.Compose(
        [
            *base_transforms,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            *base_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {"train": train_transforms, "eval": eval_transforms}


def _validate_splits() -> Tuple[float, float, float]:
    train_split = DATA_CONFIG.get("train_split", 0.8)
    val_split = DATA_CONFIG.get("val_split", 0.1)
    test_split = DATA_CONFIG.get("test_split", 0.1)
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_split, val_split, and test_split must sum to 1.0")
    return train_split, val_split, test_split


def load_datasets(
    dataset_name: str | None = None,
    data_root: str | None = None,
    seed: int = 42,
) -> DatasetBundle:
    train_split, val_split, test_split = _validate_splits()
    name, dataset_root, metadata = resolve_dataset(dataset_name, data_root)

    base_dataset = datasets.ImageFolder(dataset_root)
    if len(base_dataset) == 0:
        raise ValueError("No samples found for the selected dataset")

    transform_set = build_transforms(metadata["input_size"], metadata["grayscale"])
    total_samples = len(base_dataset)
    num_train = int(total_samples * train_split)
    num_val = int(total_samples * val_split)
    num_test = total_samples - num_train - num_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    train_dataset = datasets.ImageFolder(dataset_root, transform=transform_set["train"])
    eval_dataset = datasets.ImageFolder(dataset_root, transform=transform_set["eval"])

    return DatasetBundle(
        train=Subset(train_dataset, train_indices),
        val=Subset(eval_dataset, val_indices),
        test=Subset(eval_dataset, test_indices),
        class_names=tuple(base_dataset.classes),
        dataset_name=name,
    )


def create_dataloaders(
    dataset_name: str | None = None,
    data_root: str | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> Tuple[Dict[str, DataLoader], Tuple[str, ...], str]:
    datasets_bundle = load_datasets(dataset_name=dataset_name, data_root=data_root, seed=seed)
    effective_batch_size = batch_size or TRAINING_CONFIG["batch_size"]

    loaders = {
        "train": DataLoader(
            datasets_bundle.train,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets_bundle.val,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets_bundle.test,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders, datasets_bundle.class_names, datasets_bundle.dataset_name
