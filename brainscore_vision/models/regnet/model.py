import functools

import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

# these layer choices were not investigated in any depth, we blindly picked all high-level blocks
LAYERS = ['trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4']


def get_model():
    model = torchvision.models.regnet_y_400mf(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='regnet_y_400mf', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
