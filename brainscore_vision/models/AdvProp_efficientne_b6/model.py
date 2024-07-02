import functools

from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from efficientnet_pytorch import EfficientNet

"""
Template module for a base model submission to brain-score
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    model = name
    AT = False
    if 'AdvProp_' in name:
        AT = True
        model = model.split('AdvProp_')[-1]

    model = EfficientNet.from_pretrained(model, advprop=AT)
    model.set_swish(memory_efficient=False)

    if AT:
        preprocessing = functools.partial(load_preprocess_images, image_size=224, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    else:
        preprocessing = functools.partial(load_preprocess_images, image_size=224)

    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    from types import MethodType
    def _output_layer(self):
        return self._model._fc

    wrapper._output_layer = MethodType(_output_layer, wrapper)
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
    lmap = {
        'efficientnet-b6' : [f'_blocks.{i}' for i in range(45)]
    }
    name = name.split('AdvProp_')[-1]
    assert name in lmap
    return lmap[name]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
