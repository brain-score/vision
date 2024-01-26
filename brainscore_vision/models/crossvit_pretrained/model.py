# -*- coding: utf-8 -*-
import functools
import ssl

from timm.models import create_model
import torch
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_images,
    preprocess_images,
)
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

BIBTEX = ""
LAYERS = [
    "blocks.1.blocks.1.0.norm1",
    "blocks.1.blocks.1.4.norm2",
    "blocks.1.blocks.1.0.mlp.act",
    "blocks.2.revert_projs.1.2",
]
# Description of Layers:
# Behavior : 'blocks.2.revert_projs.1.2'
# IT       : 'blocks.1.blocks.1.4.norm2'
# V1       : 'blocks.1.blocks.1.0.norm1'
# V2       : 'blocks.1.blocks.1.0.mlp.act'
# V4       : 'blocks.1.blocks.1.0.mlp.act'
INPUT_SIZE = 256


def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def get_model():
    # Generate Model
    model = create_model("crossvit_18_dagger_408", pretrained=True)
    model.eval()

    # Load Model and create necessary methods
    # init the model and the preprocessing:
    preprocessing = functools.partial(
        load_preprocess_custom_model, image_size=224)
    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier="cv_18_dagger_408_pretrained", model=model, preprocessing=preprocessing
    )

    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
