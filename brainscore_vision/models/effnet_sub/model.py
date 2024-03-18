import functools

import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import numpy as np
import timm
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
from brainscore_vision.model_helpers.check_submission import check_models
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

import os

image_size = 224
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


def custom_image_preprocess(images, **kwargs):

    transforms_val = Compose([
        Resize(image_size, image_size),
        Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()])

    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)

    return images


def load_preprocess_images_custom(
        image_filepaths, preprocess_images=custom_image_preprocess, **kwargs):
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
    return ['effnetb0_retrain']


class EffNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.efnet_model = timm.create_model(
            'tf_efficientnet_b0_ns', pretrained=True)

    def forward(self, x):
        x = self.efnet_model.classifier(x)
        return x


def get_model():
    model_tf_efficientnet_b0_ns = EffNet()
    download_weights(
        bucket='brainscore-vision',
        folder_path='models/effnet_sub',
        filename_version_sha=[
            ('contenttf_efficientnet_b0_ns_epoch1_best_score0.49088030619938183.pth',
             'v1QmuIZpiIt0No6xhzi.JhhkEVA3ky9f',
             '051291ff100dfd48478cf4d8c7b960f59cae807c')],
        save_directory=Path(__file__).parent)
    model_tf_efficientnet_b0_ns.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "contenttf_efficientnet_b0_ns_epoch1_best_score0.49088030619938183.pth"),
            )["model"])
    model = model_tf_efficientnet_b0_ns.efnet_model

    preprocessing = functools.partial(load_preprocess_images_custom,
                                      preprocess_images=custom_image_preprocess,
                                      )

    wrapper = PytorchWrapper(
        identifier=name,
        model=model,
        preprocessing=preprocessing,
        batch_size=8)

    wrapper.image_size = image_size
    return wrapper


LAYERS = ["blocks.0.0", "blocks.1.1", "blocks.2.0", "blocks.2.1", "blocks.3.0", "blocks.3.1", "blocks.3.2",
          "blocks.4.0", "blocks.4.1", "blocks.4.2", "blocks.5.0", "blocks.5.1", "blocks.5.2", "blocks.5.3", "blocks.6.0",
          "global_pool", "global_pool.flatten", "global_pool.pool"]

BIBTEX = """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
