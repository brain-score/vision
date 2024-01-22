import functools

import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_image
import numpy as np
import timm
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize, CenterCrop
)
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models

import os

BIBTEX = """"""
LAYERS = ["blocks.0.0", "blocks.1.1", "blocks.2.0", "blocks.2.1", "blocks.3.0", "blocks.3.1", "blocks.3.2",
          "blocks.4.0", "blocks.4.1", "blocks.4.2", "blocks.5.0", "blocks.5.1", "blocks.5.2", "blocks.5.3", "blocks.6.0",
          "global_pool", "global_pool.flatten", "global_pool.pool"]


image_resize = 256
image_crop = 224
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


def custom_image_preprocess(images, **kwargs):
    transforms_val = Compose([
        Resize(image_resize, image_resize),
        CenterCrop(image_crop, image_crop),
        Normalize(mean=norm_mean, std=norm_std,),
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


class EffNetBX(nn.Module):
    def __init__(self,):
        super().__init__()
        self.efnet_model = timm.create_model(
            'tf_efficientnet_b0_ns', pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_tf_efficientnet_b0_ns = EffNetBX()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    download_weights(
        bucket='brainscore-vision',
        folder_path='models/effnetretrain_cutmix_augmix_sub',
        filename_version_sha=[
            ('tf_efficientnet_b0_ns_cutmix_augmix_epoch1_score0.5112402985466626_best.pth',
             'xR4.l6Nx6TD3xs.J2sCOqyQe5f_8xPC0',
             '54ac117c484f94fd0264d4dd9ded4b7aa70e5e8c')],
        save_directory=Path(__file__).parent)

    model_tf_efficientnet_b0_ns.load_state_dict(
        torch.load(
            dir_path +
            "/tf_efficientnet_b0_ns_cutmix_augmix_epoch1_score0.5112402985466626_best.pth",
            map_location=device)["model"])
    model = model_tf_efficientnet_b0_ns.efnet_model

    preprocessing = functools.partial(load_preprocess_images_custom,
                                      preprocess_images=custom_image_preprocess,
                                      )

    wrapper = PytorchWrapper(
        identifier="effnetb0_cutmix_augmix_epoch1",
        model=model,
        preprocessing=preprocessing,
        batch_size=8)

    wrapper.image_size = image_crop
    return wrapper


if __name__ == '__main__':
    check_models.check_base_models(__name__)
