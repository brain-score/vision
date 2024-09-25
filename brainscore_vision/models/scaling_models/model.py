import os
import functools
import json
from pathlib import Path
import ssl

import torchvision.models
import torch

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

import timm
import numpy as np
import torchvision.transforms as T
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    resize_size: int,
    crop_size: int,
    interpolation: str,
    transforms=None,
):
    if transforms is None:
        interpolation = get_interpolation_mode(interpolation)
        transforms = A.Compose(
            [
                A.Resize(resize_size, resize_size, p=1.0, interpolation=interpolation),
                A.CenterCrop(crop_size, crop_size, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    if isinstance(transforms, T.Compose):
        images = [transforms(image) for image in images]
        images = [np.array(image) for image in images]
        images = np.stack(images)
    elif isinstance(transforms, A.Compose):
        images = [transforms(image=np.array(image))["image"] for image in images]
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
    resize_size = config["resize_size"]
    crop_size = config["crop_size"]
    interpolation = config["interpolation"]
    num_classes = config["num_classes"]
    ckpt_url = config["checkpoint_url"]
    use_timm = config["use_timm"]
    timm_model_name = config["timm_model_name"]
    epoch = config["epoch"]
    load_model_ema = config["load_model_ema"]
    output_head = config["output_head"]
    is_vit = config["is_vit"]
    
    # Temporary fix for vit models
    # See https://github.com/brain-score/vision/pull/1232
    if is_vit:
        os.environ['RESULTCACHING_DISABLE'] = 'brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored'

    
    # Initialize model
    if use_timm:
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
    else:
        model = eval(f"torchvision.models.{model_name}(weights=None)")
        if num_classes != 1000:
            exec(f'''{output_head} = torch.nn.Linear(
                in_features={output_head}.in_features,
                out_features=num_classes,
                bias={output_head}.bias is not None,
                )'''
            )

    # Load model weights
    state_dict = torch.hub.load_state_dict_from_url(
        ckpt_url,
        check_hash=True,
        file_name=f"{model_id}_ep{epoch}.pt",
        map_location="cpu",
    )
    if load_model_ema:
        state_dict = state_dict["state"]["model_ema_state_dict"]
    else:
        state_dict = state_dict["state"]["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded from {ckpt_url}")

    # Wrap model
    preprocessing = functools.partial(
        load_preprocess_images_custom,
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        transforms=None
    )
    wrapper = PytorchWrapper(
        identifier=model_id, model=model, preprocessing=preprocessing
    )
    return wrapper
