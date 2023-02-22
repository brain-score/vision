from model_tools.check_submission import check_models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.brain_transformation import TemporalIgnore, ModelCommitment
import torchvision
from torchvision.models import resnet50

import functools
import torch
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

"""
Template module for a base model submission to brain-score
"""
model = torchvision.models.resnet50()
pt_model_path = os.path.abspath(os.path.join(dir_path, 'resnet50_vito.pth'))
pt_class_path = os.path.abspath(os.path.join(dir_path, 'checkpointrrc4.pth.tar'))
r50_vito = torch.load(pt_model_path, map_location='cpu').state_dict()
classifier = torch.load(pt_class_path, map_location='cpu')['state_dict']
r50_vito['fc.weight'] = classifier['module.linear.weight']
r50_vito['fc.bias'] = classifier['module.linear.bias']
model.load_state_dict(r50_vito, strict=False)
model.eval()
preprocessing = functools.partial(load_preprocess_images, image_size=224)
wrapper = PytorchWrapper(identifier='resnet50-vito', model=model, preprocessing=preprocessing)
wrapper.image_size = 224
vito_brain_model = ModelCommitment('resnet50-vito',activations_model=wrapper,layers=['layer1.2', 'layer2.3', 'layer3.5', 'layer4.2'],
region_layer_map={'V1': 'layer1.2', 'V2': 'layer2.3', 'V4': 'layer3.5', 'IT': 'layer4.2'}, visual_degrees=12)

def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['resnet50-vito12']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet50-vito12'
    model = vito_brain_model
    return model



def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_brain_models(__name__)
