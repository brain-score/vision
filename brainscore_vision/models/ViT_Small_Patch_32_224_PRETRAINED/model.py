# -*- coding: utf-8 -*-
import functools
import ssl

from timm.models import create_model

from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_images,
    preprocess_images,
)
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

BIBTEX = ""

INPUT_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 64

model_identifier_base = (
    f"ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-{INPUT_SIZE}-CROP-SIZE-{CROP_SIZE}"
)
MODEL_IDENTIFIERS = {
    f"{model_identifier_base}-{region}" for region in {"V1", "V2", "V4", "IT"}
}

# Description of Layers:
# Behavior : pre_logits
# IT       : ['blocks.7.norm2']
# V1       : ['blocks.1.mlp.act']
# V2       : ['blocks.10.norm1']
# V4       : ['blocks.2.mlp.act']
region_layers = {
    "V1": ["blocks.1.mlp.act"],
    "V2": ["blocks.10.norm1"],
    "V4": ["blocks.2.mlp.act"],
    "IT": ["blocks.7.norm2"],
}
LAYERS = {
    f"{model_identifier_base}-{region}": region_layers[region]
    for region in region_layers.keys()
}


def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def get_model(identifier):
    assert identifier in MODEL_IDENTIFIERS

    # Generate Model
    model = create_model("vit_small_patch32_224", pretrained=True)
    model.eval()

    ### Load Model and create necessary methods
    # init the model and the preprocessing:
    preprocessing = functools.partial(
        load_preprocess_custom_model, image_size=CROP_SIZE
    )
    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=model,
        preprocessing=preprocessing,
        batch_size=BATCH_SIZE,
    )
    wrapper = activations_model
    wrapper.image_size = CROP_SIZE
    return wrapper


if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
