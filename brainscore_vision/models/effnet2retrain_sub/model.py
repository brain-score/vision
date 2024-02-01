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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


BIBTEX = """"""
LAYERS = ['blocks.0.0', 'blocks.0.1',
          'blocks.1.0', 'blocks.1.1', 'blocks.1.2', 'blocks.1.3',
          'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 'blocks.2.3',
          'blocks.3.0', 'blocks.3.1', 'blocks.3.2', 'blocks.3.3', 'blocks.3.4', 'blocks.3.5',
          'blocks.4.0', 'blocks.4.1', 'blocks.4.2', 'blocks.4.3', 'blocks.4.4', 'blocks.4.5', 'blocks.4.6', 'blocks.4.7', 'blocks.4.8',
          'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.5.3', 'blocks.5.4', 'blocks.5.5', 'blocks.5.6', 'blocks.5.7',
          'blocks.5.8', 'blocks.5.9', 'blocks.5.10', 'blocks.5.11', 'blocks.5.12', 'blocks.5.13', 'blocks.5.14',
          "conv_head", "global_pool", "global_pool.flatten", "global_pool.pool"]


class CFG:
    size_resize = 300
    size = 300
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]


def custom_image_preprocess(images, **kwargs):
    transforms_val = Compose([
        Resize(CFG.size_resize, CFG.size_resize),
        CenterCrop(CFG.size, CFG.size),
        Normalize(mean=CFG.norm_mean, std=CFG.norm_std,),
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


class EffNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.efnet_model = timm.create_model(
            "tf_efficientnetv2_s_in21ft1k", pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)

        return x


def get_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    download_weights(
        bucket='brainscore-vision',
        folder_path='models/effnet2retrain_sub',
        filename_version_sha=[
            ('tf_efficientnetv2_s_in21ft1k_epoch0_score0.5156664923762149_best.pth',
             'q0QAedXnfXmRz6qziyJNKpZPlo4ffm1d',
             '3201c82a4cd69bb5d070a6148ab1a96ccaa2a95a')],
        save_directory=Path(__file__).parent)

    model = EffNet2()
    model.load_state_dict(
        torch.load(
            dir_path +
            "/tf_efficientnetv2_s_in21ft1k_epoch0_score0.5156664923762149_best.pth"
        )["model"])
    model = model.efnet_model

    preprocessing = functools.partial(load_preprocess_images_custom,
                                      preprocess_images=custom_image_preprocess
                                      )

    wrapper = PytorchWrapper(
        identifier="effnetv2retrain",
        model=model,
        preprocessing=preprocessing,
        batch_size=8)

    wrapper.image_size = CFG.size
    return wrapper


if __name__ == '__main__':
    check_models.check_base_models(__name__)
