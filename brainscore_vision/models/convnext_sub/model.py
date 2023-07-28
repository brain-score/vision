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
    Compose, Normalize, Resize,CenterCrop
    )
from albumentations.pytorch import ToTensorV2
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

import os 

image_resize = 256
image_crop = 224
norm_mean = [0.485, 0.456, 0.406] 
norm_std = [0.229, 0.224, 0.225]

def custom_image_preprocess(images, **kwargs):
    
    transforms_val = Compose([
        Resize(image_resize, image_resize),
        CenterCrop(image_crop, image_crop),
        Normalize(mean=norm_mean,std=norm_std,),
        ToTensorV2()])
    
    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)

    return images

def load_preprocess_images_custom(image_filepaths, preprocess_images=custom_image_preprocess,  **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
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
    return ['convnext_base_in22ft1k_256x224']



def get_model(name):
    assert name == 'convnext_base_in22ft1k_256x224'
    model=  timm.create_model('convnext_base_in22ft1k', pretrained=True)
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #model_tf_efficientnet_b1_ns.load_state_dict(torch.load(dir_path + "/tf_efficientnet_b1_ns_cutmix_augmix_SAM_epoch1_5_avg.pth", map_location=torch.device('cpu'))["model"])
    #model = model_tf_efficientnet_b1_ns.efnet_model
    filter_elems = set(["downsample", "norm", "mlp", "act", "conv_dw", "drop_path"])
    layer_list = [layer for layer, _ in model.named_modules() if not any(i in layer for i in filter_elems)]
    print(layer_list)
    print(len(layer_list))

    
    preprocessing = functools.partial(load_preprocess_images_custom, 
                                        preprocess_images=custom_image_preprocess,
                                        )


    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing, batch_size=8)

    wrapper.image_size = image_crop
    return wrapper


def get_layers(name):
    assert name == 'convnext_base_in22ft1k_256x224'
    return [
        'stem', 'stem.0', 'stem.1', 
        'stages', 'stages.0', 'stages.0.blocks', 
        'stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2', 
        'stages.1', 'stages.1.blocks', 'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2', 
        'stages.2', 'stages.2.blocks', 'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2', 'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5', 'stages.2.blocks.6', 
        'stages.2.blocks.7', 'stages.2.blocks.8', 'stages.2.blocks.9', 'stages.2.blocks.10', 'stages.2.blocks.11', 'stages.2.blocks.12', 'stages.2.blocks.13', 'stages.2.blocks.14', 
        'stages.2.blocks.15', 'stages.2.blocks.16', 'stages.2.blocks.17', 'stages.2.blocks.18', 'stages.2.blocks.19', 'stages.2.blocks.20', 'stages.2.blocks.21', 'stages.2.blocks.22', 
        'stages.2.blocks.23', 'stages.2.blocks.24', 'stages.2.blocks.25', 'stages.2.blocks.26', 'stages.3', 'stages.3.blocks', 'stages.3.blocks.0', 'stages.3.blocks.1', 
        'stages.3.blocks.2', 'head', 'head.global_pool', 'head.global_pool.flatten', 'head.global_pool.pool'
    ]

def get_bibtex(model_identifier):
    return """@Article{liu2022convnet,
                author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
                title   = {A ConvNet for the 2020s},
                journal = {arXiv preprint arXiv:2201.03545},
                year    = {2022},
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
