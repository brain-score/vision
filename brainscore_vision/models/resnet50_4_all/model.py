
import os
import functools
import json
from pathlib import Path
import ssl

import torchvision.models
import torch
import gdown

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

import numpy as np
import torchvision.transforms as T
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from brainscore_vision.model_helpers.s3 import load_weight_file
# Disable SSL verification 
ssl._create_default_https_context = ssl._create_unverified_context


BIBTEX=""""""

with open(Path(__file__).parent / "config.json", "r") as f:
    MODEL_CONFIGS = json.load(f)
def load_config(json_file):
    # Get the directory containing this script (model.py)
    base_dir = os.path.dirname(__file__)

    # Construct the path to the JSON file
    json_path = os.path.join(base_dir, json_file)

    # Read the JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_image(image_filepath):
    return Image.open(image_filepath).convert("RGB")


def get_interpolation_mode(interpolation: str) -> int:
    #Returns the interpolation mode for albumentations
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
    model_id = config["model_id"]
    resize_size = config["resize_size"]
    crop_size = config["crop_size"]
    interpolation = config["interpolation"]
    is_vit = config["is_vit"]
    keyword = config["keyword"]
    network = config["network"]
    identifier = config["model_id"]
    # Unpack model config
    # Temporary fix for vit models
    # See https://github.com/brain-score/vision/pull/1232
    if is_vit:
        os.environ['RESULTCACHING_DISABLE'] = 'brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored'

    ckpt_url = "https://drive.google.com/file/d/1K7GcuEsvHBzVT2T7ONqPDJrNqpCTH-K3/view?usp=share_link"
    output = 'checkpoint.ckpt'
    gdown.download('checkpoint.ckpt',output)
    if keyword != 'imagenet_trained' and keyword != 'no_training':
        lx_whole = [f"checkpoint.ckpt"]
        if len(lx_whole) > 1:
            lx_whole = [lx_whole[-1]]
    elif keyword == 'imagenet_trained' or keyword == 'no_training':
        print('keyword is imagenet')
        lx_whole = ['x']  
    model = torch.hub.load('pytorch/vision', network, pretrained=False)
    ckpt = torch.load(output, map_location='cpu')
    if keyword == 'imagenet_trained' or keyword=='no_training':
        model_ckpt = 'x'
    else: 
        model_ckpt = None
    last_module_name = None
    last_module = None
    layers = []
    if keyword == 'imagenet_trained' and network != 'clip':
        model = torch.hub.load('pytorch/vision', network, pretrained=True)
        for name, module in model.named_modules():
            last_module_name = name
            last_module = module
            layers.append(name)
    if keyword == 'no_training' and network!='clip':
        model = torch.hub.load('pytorch/vision', network, pretrained=False)
    else:
        model = torch.hub.load('pytorch/vision', network, pretrained=False)
    if model_ckpt != 'x':
        ckpt = torch.load(weights_path, map_location='cpu')
    if model_ckpt != 'x' and network == 'alexnet' and keyword != 'imagenet_trained':
        ckpt2 = {}
        for keys in ckpt['state_dict']:
            print(keys)
            print(ckpt['state_dict'][keys].shape)
            print('---')
            k2 = keys.split('model.')[1]
            ckpt2[k2] = ckpt['state_dict'][keys]
        model.load_state_dict(ckpt2)
    if model_ckpt != 'x' and network == 'vgg16' and keyword != 'imagenet_trained':
        ckpt2 = {}
        for keys in ckpt['state_dict']:
            print(keys)
            print(ckpt['state_dict'][keys].shape)
            print('---')
            k2 = keys.split('model.')[1]
            ckpt2[k2] = ckpt['state_dict'][keys]
        model.load_state_dict(ckpt2)
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
