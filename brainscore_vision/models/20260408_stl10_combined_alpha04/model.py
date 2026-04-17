from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models


MODEL_IDENTIFIER = '20260408_stl10_combined_alpha04'
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pth"
IMAGE_SIZE = 96
NORMALIZE_MEAN = (0.44671062, 0.43980984, 0.40664646)
NORMALIZE_STD = (0.26034098, 0.25657727, 0.27126738)
FEATURE_VARIANT = 'normalized'
BIBTEX = 'xx'


class L2Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)


class SubmissionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, _ = build_resnet_encoder()
        self.embedding = nn.Identity()
        self.normalized = L2Normalize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.embedding(x)
        if FEATURE_VARIANT == "normalized":
            x = self.normalized(x)
        return x


def build_resnet_encoder() -> tuple[nn.Module, int]:
    encoder = models.resnet18(weights=None)
    in_features = encoder.fc.in_features
    encoder.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder, in_features


def load_checkpoint_state(checkpoint_path: Path):
    try:
        payload = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def extract_encoder_state(state_dict):
    if any(key.startswith("encoder.") for key in state_dict):
        return {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
    return state_dict


def get_model_list():
    return [MODEL_IDENTIFIER]


def get_model(name):
    assert name == MODEL_IDENTIFIER
    model = SubmissionBackbone()
    state_dict = load_checkpoint_state(WEIGHTS_PATH)
    encoder_state = extract_encoder_state(state_dict)
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not load cleanly: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=IMAGE_SIZE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
    )
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == MODEL_IDENTIFIER
    return ['normalized']


def get_bibtex(model_identifier):
    assert model_identifier == MODEL_IDENTIFIER
    return BIBTEX


if __name__ == "__main__":
    check_models.check_base_models(__name__)
