from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Tuple
import json

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformedSubset(Dataset):
    """Wrap a subset so each split can use its own transform."""

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return train, validation, and test transforms."""
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "test": transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }


def build_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    num_workers: int = 4,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> Tuple[Dict[str, DataLoader], Tuple[int, int, int], list[str]]:
    """Load the dataset and return train/validation/test DataLoaders."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    if val_split + test_split >= 1.0:
        raise ValueError("Validation split and test split must sum to less than 1.0.")

    transforms_map = get_transforms(image_size=image_size)
    dataset = datasets.ImageFolder(root, transform=None)
    total = len(dataset)
    if total == 0:
        raise ValueError(f"No images found in dataset root: {root}")

    val_size = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Training split is zero or negative. Adjust val_split and test_split.")
    split_path = Path(split_file) if split_file is not None else None
    if split_path is not None and split_path.exists():
        with open(split_path, "r", encoding="utf-8") as fh:
            indices = json.load(fh)
        train_subset = Subset(dataset, indices["train"])
        val_subset = Subset(dataset, indices["val"])
        test_subset = Subset(dataset, indices["test"])
    else:
        lengths = [train_size, val_size, test_size]
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset, test_subset = random_split(dataset, lengths, generator=generator)
        if split_path is not None:
            indices = {
                "train": list(getattr(train_subset, "indices", [])),
                "val": list(getattr(val_subset, "indices", [])),
                "test": list(getattr(test_subset, "indices", [])),
            }
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as fh:
                json.dump(indices, fh)

    train_dataset = TransformedSubset(train_subset, transforms_map["train"])
    val_dataset = TransformedSubset(val_subset, transforms_map["val"])
    test_dataset = TransformedSubset(test_subset, transforms_map["test"])

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }

    return loaders, (train_size, val_size, test_size), dataset.classes


def get_class_counts(data_dir: str | Path) -> Counter:
    """Return the class distribution for the raw image folders."""
    root = Path(data_dir)
    dataset = datasets.ImageFolder(root, transform=None)
    return Counter(dataset.targets)
