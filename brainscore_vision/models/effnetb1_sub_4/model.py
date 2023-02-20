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

image_resize = 324
image_crop = 288
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
    return ['effnetb1_cutmix_augmix_epoch4_324x288_notoomanylayers']


class EffNetBX(nn.Module):
    def __init__(self,):
        super().__init__ ()
        self.efnet_model = timm.create_model('tf_efficientnet_b1_ns', pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x

def get_model(name):
    assert name == 'effnetb1_cutmix_augmix_epoch4_324x288_notoomanylayers'
    model_tf_efficientnet_b1_ns= EffNetBX()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_tf_efficientnet_b1_ns.load_state_dict(torch.load(dir_path + "/tf_efficientnet_b1_ns_cutmix_augmix_epoch4_score0.5138311463188558_best.pth", map_location=torch.device('cpu'))["model"])
    model = model_tf_efficientnet_b1_ns.efnet_model
    filter_elems = set(["se", "act", "bn"])
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
    assert name == 'effnetb1_cutmix_augmix_epoch4_324x288_notoomanylayers'
    return ['conv_stem', 'blocks', 'blocks.0', 'blocks.0.0', 'blocks.0.0.conv_dw', 'blocks.0.0.conv_pw', 'blocks.0.1', 'blocks.0.1.conv_dw', 'blocks.0.1.conv_pw', 
    'blocks.1', 'blocks.1.0', 'blocks.1.0.conv_pw', 'blocks.1.0.conv_dw', 'blocks.1.0.conv_pwl', 'blocks.1.1', 'blocks.1.1.conv_pw', 'blocks.1.1.conv_dw', 'blocks.1.1.conv_pwl', 
    'blocks.1.2', 'blocks.1.2.conv_pw', 'blocks.1.2.conv_dw', 'blocks.1.2.conv_pwl', 'blocks.2', 'blocks.2.0', 'blocks.2.0.conv_pw', 'blocks.2.0.conv_dw', 'blocks.2.0.conv_pwl', 
    'blocks.2.1', 'blocks.2.1.conv_pw', 'blocks.2.1.conv_dw', 'blocks.2.1.conv_pwl', 'blocks.2.2', 'blocks.2.2.conv_pw', 'blocks.2.2.conv_dw', 'blocks.2.2.conv_pwl', 'blocks.3', 
    'blocks.3.0', 'blocks.3.0.conv_pw', 'blocks.3.0.conv_dw', 'blocks.3.0.conv_pwl', 'blocks.3.1', 'blocks.3.1.conv_pw', 'blocks.3.1.conv_dw', 'blocks.3.1.conv_pwl', 'blocks.3.2', 
    'blocks.3.2.conv_pw', 'blocks.3.2.conv_dw', 'blocks.3.2.conv_pwl', 'blocks.3.3', 'blocks.3.3.conv_pw', 'blocks.3.3.conv_dw', 'blocks.3.3.conv_pwl', 'blocks.4', 'blocks.4.0', 
    'blocks.4.0.conv_pw', 'blocks.4.0.conv_dw', 'blocks.4.0.conv_pwl', 'blocks.4.1', 'blocks.4.1.conv_pw', 'blocks.4.1.conv_dw', 'blocks.4.1.conv_pwl', 'blocks.4.2', 
    'blocks.4.2.conv_pw', 'blocks.4.2.conv_dw', 'blocks.4.2.conv_pwl', 'blocks.4.3', 'blocks.4.3.conv_pw', 'blocks.4.3.conv_dw', 'blocks.4.3.conv_pwl', 'blocks.5', 
    'blocks.5.0', 'blocks.5.0.conv_pw', 'blocks.5.0.conv_dw', 'blocks.5.0.conv_pwl', 'blocks.5.1', 'blocks.5.1.conv_pw', 'blocks.5.1.conv_dw', 'blocks.5.1.conv_pwl', 
    'blocks.5.2', 'blocks.5.2.conv_pw', 'blocks.5.2.conv_dw', 'blocks.5.2.conv_pwl', 'blocks.5.3', 'blocks.5.3.conv_pw', 'blocks.5.3.conv_dw', 'blocks.5.3.conv_pwl', 
    'blocks.5.4', 'blocks.5.4.conv_pw', 'blocks.5.4.conv_dw', 'blocks.5.4.conv_pwl', 'blocks.6', 'blocks.6.0', 'blocks.6.0.conv_pw', 'blocks.6.0.conv_dw', 
    'blocks.6.0.conv_pwl', 'blocks.6.1', 'blocks.6.1.conv_pw', 'blocks.6.1.conv_dw', 'blocks.6.1.conv_pwl', 'conv_head', 'global_pool', 'global_pool.flatten', 
    'global_pool.pool', 'classifier']

def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
