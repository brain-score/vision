import functools

import torch

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

BIBTEX = """@article{oquab2023dinov2,
  title={Dinov2: Learning robust visual features without supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}"""

def get_model_list():
    return ['dinov2_vits14_reg_linear']

def net_constructors(net):
    if net == "dinov2_vits14_reg_linear":
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
    else:
        raise ValueError(f"Could not find DINOV2 network with registers: {net}")

def get_layers(net):
    model = net_constructors(net)
    layers  = []
    i = 0
    while hasattr(model.backbone.blocks, str(i)):
        layers.append(f'backbone.blocks.{i}')
        i += 1
    assert hasattr(model, 'linear_head')
    layers.append('linear_head')
    return layers

def get_model(net):
    model = net_constructors(net)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier=net,
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_bibtex(model_identifier):
    return BIBTEX

if __name__ == '__main__':
    check_models.check_base_models(__name__)
