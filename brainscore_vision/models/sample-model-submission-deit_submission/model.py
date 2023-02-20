import torch
import timm
assert timm.__version__ == "0.3.2"
import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    return ['deit']


def get_model(name):
    assert name == 'deit'
    with torch.no_grad():
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='deit', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'deit'
    num_of_blocks = 12
    return [f'blocks.{i}' for i in range(num_of_blocks)]
    #return [f'blocks.{i}.mlp.fc2' for i in range(num_of_blocks)]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@article{touvron2020deit,
                    title={Training data-efficient image transformers & distillation through attention},
                    author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
                    journal={arXiv preprint arXiv:2012.12877},
                    year={2020}}"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
    
