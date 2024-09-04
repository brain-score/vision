from brainscore_vision.model_helpers.check_submission import check_models
import functools
import os
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
import torch

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models


def get_model_list():
    return ['alexnet_7be5be79_relus']


def get_model(name):
    assert name == 'alexnet_7be5be79_relus'
    model = torchvision.models.alexnet(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet_7be5be79_relus', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'alexnet_7be5be79_relus'
    return ['features.1','features.4', 'features.7', 'features.9', 'features.11', 'classifier.2', 'classifier.5', 'classifier.6']


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
