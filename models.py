"""Transfer-learning model factory (torchvision pretrained backbones)."""
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tv

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def get_transfer_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, int]:
    """Return (model, required_input_size) with a fresh classification head.

    Supported backbones: "resnet18", "mobilenet_v3_small".
    """
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = tv.ResNet18_Weights.DEFAULT
        model = tv.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        input_size = 224
    elif model_name == "mobilenet_v3_small":
        weights = tv.MobileNet_V3_Small_Weights.DEFAULT
        model = tv.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        input_size = 224
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name!r}. Use 'resnet18' or 'mobilenet_v3_small'."
        )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model, input_size