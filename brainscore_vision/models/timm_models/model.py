import os
import functools
import json
from pathlib import Path
import ssl

import timm
import numpy as np
import torchvision.transforms as T
from PIL import Image

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

# Disable SSL verification 
ssl._create_default_https_context = ssl._create_unverified_context

BIBTEX = """"""


with open(Path(__file__).parent / "model_configs.json", "r") as f:
    MODEL_CONFIGS = json.load(f)


def load_image(image_filepath):
    return Image.open(image_filepath).convert("RGB")


def get_interpolation_mode(interpolation: str) -> int:
    """Returns the interpolation mode for albumentations"""
    if "linear" or "bilinear" in interpolation:
        return 1
    elif "cubic" or "bicubic" in interpolation:
        return 2
    else:
        raise NotImplementedError(f"Interpolation mode {interpolation} not implemented")


def custom_image_preprocess(
    images,
    transforms=T.Compose,
):
    if isinstance(transforms, T.Compose):
        images = [transforms(image) for image in images]
        images = [np.array(image) for image in images]
        images = np.stack(images)
    else:
        raise NotImplementedError(
            f"Transform of type {type(transforms)} is not implemented"
        )

    return images


def load_preprocess_images_custom(
    image_filepaths, preprocess_images=custom_image_preprocess, **kwargs
):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images


def get_model(model_id:str):
    # Unpack model config
    config = MODEL_CONFIGS[model_id]
    model_name = config["model_name"]
    model_id = config["model_id"]
    timm_model_name = config["timm_model_name"]
    is_vit = config["is_vit"]
    
    # Temporary fix for vit models
    # See https://github.com/brain-score/vision/pull/1232
    if is_vit:
        os.environ['RESULTCACHING_DISABLE'] = 'brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored'

    
    # Initialize model
    model = timm.create_model(timm_model_name, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(f"Model {model_name} loaded")

    # Wrap model
    preprocessing = functools.partial(
        load_preprocess_images_custom,
        transforms=transforms
    )
    wrapper = PytorchWrapper(
        identifier=model_id, model=model, preprocessing=preprocessing
    )
    return wrapper
