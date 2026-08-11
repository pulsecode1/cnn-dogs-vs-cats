from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import build_data_loaders
from model import create_model


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model on the Cats vs Dogs test set.")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"), help="Root dataset folder.")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights file path.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory to save evaluation artifacts.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Proportion of the dataset used for validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Proportion of the dataset used for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional path to a split indices JSON file to use for deterministic splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine split file: prefer explicit --split-file, else use output_dir/splits.json
    split_file = args.split_file if args.split_file is not None else args.output_dir / "splits.json"

    loaders, splits, class_names = build_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_file=split_file,
    )
    test_loader = loaders["test"]

    model = create_model(num_classes=len(class_names), feature_extract=False, use_pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("Evaluation Results")
    print("------------------")
    print(report)

    plot_confusion_matrix(cm, class_names, args.output_dir / "confusion_matrix.png")
    with open(args.output_dir / "evaluation_report.txt", "w", encoding="utf-8") as file:
        file.write("Confusion Matrix:\n")
        file.write(str(cm) + "\n\n")
        file.write(report)

    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved evaluation report to {args.output_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
