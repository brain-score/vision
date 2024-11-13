import os
import functools
from importlib import import_module
from pathlib import Path
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import cornet
from brainscore_vision.model_helpers.s3 import load_weight_file

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    assert name == 'BT_CORnet-S-100'
    
    model = getattr(cornet, f'cornet_s')
    model = model(pretrained=True, map_location=torch.device('cpu'))
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="braintree-models/weights/epoch_100.pth.tar",
                                    version_id="dMNYhC.9bg39SrPZ2bqmUC_5rdd3P5ax",
                                    sha1="4d8c9639fca21234cd189db9db81b3f41f3afcf3")
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.module
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(model=model, preprocessing=preprocessing, identifier=name)
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
    # return the same layers for all models, as they are all resnet50s
    assert name == 'BT_CORnet-S-100'
    return ['V1','V2','V4','IT']

if __name__ == '__main__':
    check_models.check_base_models(__name__)
