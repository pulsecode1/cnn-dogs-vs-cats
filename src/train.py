from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model, count_trainable_parameters
from data_loader import build_data_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN classifier using transfer learning.")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"), help="Root dataset folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory to save model and logs.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train/val.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Proportion of the dataset used for validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Proportion of the dataset used for testing.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for model inputs.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--feature-extract", action="store_true", help="Freeze backbone layers during training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def plot_history(history: dict, output_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, splits, class_names = build_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_file=args.output_dir / 'splits.json',
    )

    print(f"Dataset classes: {class_names}")
    print(f"Training/Validation/Test split: {splits}")

    model = create_model(num_classes=len(class_names), feature_extract=args.feature_extract)
    model = model.to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = args.output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    plot_history(history, args.output_dir / "training_history.png")
    print(f"Training complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
