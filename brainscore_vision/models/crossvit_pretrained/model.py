# -*- coding: utf-8 -*-
# +
# Question
# How is determined the layer for behavior benchmark

# +
from brainscore_vision.model_helpers.check_submission import check_models
import torch 
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import os
from timm.models import create_model
from brainscore_vision.model_helpers.activations.pytorch import load_images
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

INPUT_SIZE = 256
LAYERS = ['blocks.1.blocks.1.0.norm1','blocks.1.blocks.1.4.norm2','blocks.1.blocks.1.0.mlp.act','blocks.2.revert_projs.1.2']
BIBTEX = ''


print(os.path.join(os.path.dirname(__file__)))
# Description of Layers:
# Behavior : 'blocks.2.revert_projs.1.2'
# IT       : 'blocks.1.blocks.1.4.norm2'
# V1       : 'blocks.1.blocks.1.0.norm1'
# V2       : 'blocks.1.blocks.1.0.mlp.act'
# V4       : 'blocks.1.blocks.1.0.mlp.act'

# +
def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop((image_size,image_size)),
        torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


# -

# Generate Model
model = create_model('crossvit_18_dagger_408',pretrained = True)
model.eval()

### Load Model and create necessary methods
# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_custom_model, image_size=224)
# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='cv_18_dagger_408_pretrained', model=model, preprocessing=preprocessing)
# actually make the model, with the layers you want to see specified:
#model = ModelCommitment(identifier='custom_model_v1', activations_model=activations_model,layers=LAYERS)

def get_model():
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
