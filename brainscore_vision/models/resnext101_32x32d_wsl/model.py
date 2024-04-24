import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import torch.hub
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def get_model_list():
    return ['resnext101_32x32d_wsl']


def get_model(name):
    assert name == 'resnext101_32x32d_wsl'
    model_identifier = "resnext101_32x32d_wsl"
    model = torch.hub.load('facebookresearch/WSL-Images', model_identifier)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    batch_size = {8: 32, 16: 16, 32: 8, 48: 4}
    wrapper = PytorchWrapper(identifier=model_identifier, model=model, preprocessing=preprocessing,
                             batch_size=batch_size[32])
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnext101_32x32d_wsl'
    return (['conv1'] +
            # note that while relu is used multiple times, by default the last one will overwrite all previous ones
            [f"layer{block + 1}.{unit}.relu"
             for block, block_units in enumerate([3, 4, 23, 3]) for unit in range(block_units)] +
            ['avgpool'])


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)