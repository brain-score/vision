import functools

import torch
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize
    )
from albumentations.pytorch import ToTensorV2
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models


image_size = 384
#norm_mean = [0.485, 0.456, 0.406] 
#norm_std = [0.229, 0.224, 0.225]

def custom_image_preprocess(images, **kwargs):
    #norm_mean = [0.485, 0.456, 0.406] 
    #norm_std = [0.229, 0.224, 0.225]
    transforms_val = Compose([
        Resize(image_size, image_size),
        #CenterCrop(CFG.size, CFG.size),
        Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5],),
        ToTensorV2()])
    
    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)
    return images


def load_preprocess_images_custom(image_filepaths, image_size=0, transform_timm=None, **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    if transform_timm is not None:
      images = [transform_timm(image) for image in images]
      images = np.stack(images)
    else:
      images = custom_image_preprocess(images, image_size=image_size, **kwargs)
    return images


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper()\
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


def get_model_list():
    return ['effnetv2m_custom384']


def get_model(name):
    assert name == 'effnetv2m_custom384'
    model= timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True)

    
    preprocessing = functools.partial(load_preprocess_images_custom, transform_timm=None)

    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing, batch_size=8)

    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == 'effnetv2m_custom384'
    return  ['blocks.0.0', 'blocks.0.1', 
 'blocks.1.0', 'blocks.1.1', 'blocks.1.2', 'blocks.1.3', 
 'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 'blocks.2.3', 
 'blocks.3.0', 'blocks.3.1', 'blocks.3.2', 'blocks.3.3', 'blocks.3.4', 'blocks.3.5', 
 'blocks.4.0', 'blocks.4.1', 'blocks.4.2', 'blocks.4.3', 'blocks.4.4', 'blocks.4.5', 'blocks.4.6', 'blocks.4.7', 'blocks.4.8', 
 'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.5.3', 'blocks.5.4', 'blocks.5.5', 'blocks.5.6', 'blocks.5.7', 
 'blocks.5.8', 'blocks.5.9', 'blocks.5.10', 'blocks.5.11', 'blocks.5.12', 'blocks.5.13', 'blocks.5.14',
 "conv_head", "global_pool", "global_pool.flatten", "global_pool.pool"]


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
