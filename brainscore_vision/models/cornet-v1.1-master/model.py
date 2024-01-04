import functools

import torch
import torchvision.models
from model_tools.activations import pytorch

from models import cornet
from models.tests import test_models

"""
Template module for a base model submission to brain-score
"""


class PytorchWrapper(pytorch.PytorchWrapper):

    @classmethod
    def _tensor_to_numpy(cls, output):
        return output[0].cpu().data.numpy()  # only return output, not state

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['cornet-v1.1']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'cornet-v1.1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cornet.CORnet_v1(times=7, pretrained=True, map_location=device)
    preprocessing = functools.partial(pytorch.load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='cornet-v1.1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    assert name == 'cornet-v1.1'
    return ['V1', 'V2', 'V4', 'IT']


if __name__ == '__main__':
    test_models.test_base_models(__name__)
