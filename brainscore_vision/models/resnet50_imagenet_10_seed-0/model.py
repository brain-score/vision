import functools

import torchvision.models
import torch

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

import numpy as np
import torchvision.transforms as T
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

BIBTEX = """"""

MODEL_NAME = "resnet50"
MODEL_ID = "resnet50_imagenet_10_seed-0"
MODEL_COMMITMENT = {
    "region_layer_map": {
        "V1": "layer1.0.conv1",
        "V2": "layer3.5.bn3",
        "V4": "layer3.0.conv1",
        "IT": "layer4.0.relu",
    },
    "behavioral_readout_layer": "favgpool",
    "layers": ["layer1.0.conv1", "layer3.0.conv1", "layer3.5.bn3", "layer4.0.relu"]
}

RESIZE_SIZE = 256
CROP_SIZE = 224
INTERPOLATION = "bilinear"
NUM_CLASSES = 1000
EPOCH = 100
CKPT_URL = "https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints/resnet50_imagenet_10_seed-0/ep100.pt"


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
    transforms=None,
    resize_size: int = RESIZE_SIZE,
    crop_size: int = CROP_SIZE,
    interpolation: str = INTERPOLATION,
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


def get_model():
    model = torchvision.models.resnet50()
    if NUM_CLASSES != 1000:
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=NUM_CLASSES,
            bias=model.fc.bias is not None,
        )

    state_dict = torch.hub.load_state_dict_from_url(CKPT_URL, check_hash=True, file_name=f'{MODEL_NAME}_ep{EPOCH}.pt', map_location="cpu")
    state_dict = state_dict["state"]["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded from {CKPT_URL}")

    preprocessing = functools.partial(load_preprocess_images_custom, transforms=None)
    wrapper = PytorchWrapper(
        identifier=MODEL_ID, model=model, preprocessing=preprocessing
    )
    return wrapper
