import functools
import ssl

import torch

from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images)
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

BIBTEX = """"""
LAYERS = ['conv1', 'layer1.0', 'layer1.1', 'layer1.2',
          'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
          'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3',
          'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'avgpool']


def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :return: the model instance
    """
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier='resnet50-barlow',
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
