from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import functools
import torch
import os
from brainscore_vision.model_helpers.s3 import load_weight_file
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .helpers.resnet import resnet50

dir_path = os.path.dirname(os.path.realpath(__file__))

"""
Template module for a base model submission to brain-score
"""

LAYERS = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0',
                   'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2',
                   'avgpool', 'fc']

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """

    assert name == 'resnet50-VITO-8deg-cc'
    model = resnet50(attn_pool=True)

    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="resnet50-VITO-8deg-cc/resnet50_vito_linear_sdattn.pth",
                                    version_id="null",
                                    sha1="2103a1941017fdb5f58ac77e5587e7a73cd96bd8")
    r50_vito_sd = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(r50_vito_sd, strict=False)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet50-VITO-8deg-cc', model=model, preprocessing=preprocessing)
    #layers = LAYERS
    #vito_brain_model = ModelCommitment('resnet50-VITO-8deg-cc', activations_model=wrapper, layers=layers,
    #                                   visual_degrees=8)
    return wrapper


def get_layers(name):
    assert name == 'resnet50-VITO-8deg-cc'
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return LAYERS


def get_bibtex(model_identifier):

    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    check_models.check_base_models(__name__)
