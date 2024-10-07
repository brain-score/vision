from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from torchvision.models import convnext_tiny
import functools
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
Template module for a base model submission to brain-score
"""


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'convnext_tiny_sup'

    model = convnext_tiny(weights='IMAGENET1K_V1').eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='convnext_tiny_sup', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'convnext_tiny_sup'
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return ['features.0.0', 'features.0.1', 'features.1.0', 'features.1.1', 'features.1.2', 'features.2.0', 'features.2.1','features.3.0', 'features.3.1', 'features.3.2',  'features.4.0', 'features.4.1', 'features.5.0', 'features.5.1', 'features.5.2', 'features.5.3', 'features.5.4', 'features.5.5', 'features.5.6', 'features.5.7', 'features.5.8','features.6.0', 'features.6.1', 'features.7.0', 'features.7.1', 'features.7.2', 'avgpool', 'classifier.0', 'classifier.1', 'classifier.2']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
