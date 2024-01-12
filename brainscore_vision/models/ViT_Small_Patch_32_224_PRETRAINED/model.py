# -*- coding: utf-8 -*-
# +
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from timm.models import create_model
from brainscore_vision.model_helpers.activations.pytorch import load_images
import numpy as np
import ssl

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


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.CenterCrop((image_size, image_size)),
            torchvision_preprocess(**kwargs),
        ]
    )


def torchvision_preprocess(
    normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)
):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
            lambda img: img.unsqueeze(0),
        ]
    )


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
    # actually make the model, with the layers you want to see specified:
    # model = ModelCommitment(identifier='custom_model_v1', activations_model=activations_model,layers=LAYERS)

    wrapper = activations_model
    wrapper.image_size = CROP_SIZE
    return wrapper


if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
