import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

from .architecture import build_model


IDENTIFIER = "eicircuit_resnet18_abs_group_20260921"
LAYERS = [
    "features.maxpool",
    "features.layer1.0", "features.layer1.1",
    "features.layer2.0", "features.layer2.1",
    "features.layer3.0", "features.layer3.1",
    "features.layer4.0", "features.layer4.1",
]
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(TRANSFORM(image.convert("RGB")).numpy())
    return np.stack(images)


def weight_path():
    directory = Path(__file__).resolve().parent
    local = directory / "weights.pt"
    if local.is_file():
        return local
    config = json.loads((directory / "weights_config.json").read_text())
    if config["folder_name"] is None:
        raise RuntimeError(
            "Upload the weights through Brain-Score Large File Upload and configure "
            "weights_config.json before submitting this plugin. See MODEL_CARD.md."
        )
    from brainscore_core.supported_data_standards.brainio.s3 import load_file
    return load_file(**config)


def get_model(name):
    if name != IDENTIFIER:
        raise ValueError(name)
    model = build_model("ei")
    model.load_state_dict(torch.load(weight_path(), map_location="cpu", weights_only=True), strict=True)
    model.eval()
    wrapper = PytorchWrapper(
        identifier=IDENTIFIER, model=model, preprocessing=preprocess_images, batch_size=16,
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    if name != IDENTIFIER:
        raise ValueError(name)
    return list(LAYERS)


def get_bibtex(name):
    return ""
