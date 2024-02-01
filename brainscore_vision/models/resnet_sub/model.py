# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

import functools
import os
from pathlib import Path

import torch
import torchvision.models as models
from torch.nn import Module

from brainscore_vision.model_helpers import download_weights
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper, load_preprocess_images)
from brainscore_vision.model_helpers.check_submission import check_models

BIBTEX = """"""
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def load_model(modelname='resnet', checkpoint_file=None):
    if modelname == 'resnet':
        model = models.resnet18()
    else:
        raise ValueError("Architechture {} not valid.".format(modelname))

    # model was wrapped with DataParallel, so weights require `module.` prefix
    model = Wrapper(model)
    print("=> loading checkpoint '{}'".format(checkpoint_file))

    os.makedirs(
        os.path.join(
            os.path.dirname(__file__),
            'saved-weights'),
        exist_ok=True)
    download_weights(
        bucket='brainscore-vision',
        folder_path="models/resnet-lr=0.01-sub",
        filename_version_sha=[
            ("model_best.pth.tar",
             "GNoF7vOy.D39csXROrp3_n1dzWK2NFGF",
             "372d949b4fddefba2c8c484391e5e3c4fe8fc835")],
        save_directory=os.path.join(Path(__file__).parent, 'saved-weights'))

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap

    return model


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :return: the model instance
    """
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    dirname = os.path.dirname(__file__)
    weights_path = os.path.join(dirname, 'saved-weights/model_best.pth.tar')

    print(f"weights path is: {weights_path}")
    model = load_model(checkpoint_file=weights_path)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(
        identifier='resnet-lr0.001-500c',
        model=model,
        preprocessing=preprocessing)

    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
