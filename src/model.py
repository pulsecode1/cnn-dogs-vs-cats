from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 2, feature_extract: bool = True, use_pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 model for transfer learning."""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights if use_pretrained else None)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(input_features, num_classes),
    )
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
