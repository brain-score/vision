import functools
from typing import Any

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)
from brainscore_vision.model_helpers.check_submission import check_models

BIBTEX = """@incollection{,
    title = {PPP-Tech},
    author = {},
    booktitle = {},
    editor = {Oriz DoronR},
    pages = {10-11},
    year = {2020},
    publisher = {},
    url = {}
}"""

LAYERS = [
    "features.4",
    "features.5",
    "features.8",
    "features.11",
    "features.12",
    "classifier.3",
    "classifier.4",
]


class DoriNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(DoriNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=11, stride=4, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def dorinet(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model = DoriNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["dorinet"], progress=progress)
        model.load_state_dict(state_dict)
    return model


def get_model_list():
    return ["dorinet"]


def get_model(name):
    assert name == "dorinet"
    model = dorinet()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="dorinet", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper


if __name__ == "__main__":
    check_models.check_base_models(__name__)

__all__ = ["DoriNet", "dorinet"]

model_urls = {
    "dorinet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}
