# -*- coding: utf-8 -*-
from brainscore_vision.model_helpers.check_submission import check_models
import torch
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
import os
from timm.models import create_model
from brainscore_vision.model_helpers.activations.pytorch import (
    load_images,
    preprocess_images,
)
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

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
WEIGHT_PATH = os.path.join(
    os.path.dirname(__file__), "crossvit_18_dagger_408_adv_finetuned_epoch5.pt"
)


def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def get_model():
    download_weights(
        bucket="brainscore-vision",
        folder_path="models/crossvit_18_dagger_408_finetuned_v3",
        filename_version_sha=[
            (
                "crossvit_18_dagger_408_adv_finetuned_epoch5.pt",
                "yLoNLVtxWxYXPJGM5hHCENrLKQiRzJwa",
                "c769518485e352d5a2e6f3e588d6208cbad71b69",
            )
        ],
        save_directory=Path(__file__).parent,
    )
    print(os.path.join(os.path.dirname(__file__)))

    # Generate Model
    model = create_model("crossvit_18_dagger_408", pretrained=False)
    checkpoint = torch.load(WEIGHT_PATH)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    ### Load Model and create necessary methods
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_custom_model, image_size=224)
    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier="custom_model_cv_18_dagger_408_v3",
        model=model,
        preprocessing=preprocessing,
    )

    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
