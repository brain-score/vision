import functools

from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.check_submission import check_models
from efficientnet_pytorch import EfficientNet

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    prefix = 'AdvProp_'
    #prefix = ''
    models = [
        f'{prefix}efficientnet-b{i}'
        for i in [0, 2, 4, 6, 7, 8]
        #for i in [0]
    ]
    return models

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
        'efficientnet-b0' : [f'_blocks.{i}' for i in range(16)],
        'efficientnet-b2' : [f'_blocks.{i}' for i in range(23)],
        'efficientnet-b4' : [f'_blocks.{i}' for i in range(32)],
        'efficientnet-b6' : [f'_blocks.{i}' for i in range(45)],
        'efficientnet-b7' : [f'_blocks.{i}' for i in range(55)],
        'efficientnet-b8' : [f'_blocks.{i}' for i in range(61)]
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
