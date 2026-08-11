# Cat vs Dog CNN Project

This repository implements a complete image classification pipeline for the Kaggle Cats vs Dogs dataset using transfer learning.

## Project structure

- `Dataset/` - source image folders (`Cat/`, `Dog/`)
- `src/data_loader.py` - data loading, transforms, and train/validation/test splits
- `src/model.py` - pretrained ResNet18 feature extractor with a custom classifier head
- `src/train.py` - training script with best-model saving and training history logging
- `src/evaluate.py` - test evaluation, confusion matrix, and classification report generation
- `notebooks/cnn_dogs_vs_cats.ipynb` - exploratory notebook with dataset overview, training, and visualizations
- `report.md` - project documentation, findings, and conclusions

## Dataset

- Source: Kaggle Cats vs Dogs dataset
- Classes: `Cat`, `Dog`
- Total images: 24,998

## Requirements

Install the project dependencies in the virtual environment:

```powershell
.\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run training

```powershell
python .\src\train.py --data-dir Dataset --output-dir results --epochs 3 --batch-size 32
```

## Evaluate

```powershell
python .\src\evaluate.py --data-dir Dataset --weights results/best_model.pth --output-dir results
```

## Notebook

Open `notebooks/cnn_dogs_vs_cats.ipynb` in VS Code to explore dataset samples, training flows, and visualizations.
