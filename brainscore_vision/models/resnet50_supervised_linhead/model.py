"""
Supervised ResNet-50 model for Brain-Score submission.
"""

import functools
from collections import OrderedDict

import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)
from brainscore_vision.model_helpers.check_submission import check_models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

REPO_ID = "niels-c/resnet50_supervised_linhead"
FILENAME = "resnet50_supervised_linhead.safetensors"


def load_model():
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", a ResNet-50 whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    backbone = torchvision.models.resnet50(weights=None)
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


def get_model(name):
    assert name == "resnet50_supervised_linhead"
    image_size = 224
    model = load_model()  # head at backbone.fc, logits
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(
        identifier="resnet50_supervised_linhead",
        model=model,
        preprocessing=preprocessing,
    )
    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == "resnet50_supervised_linhead"
    return [
        "backbone.conv1",
        "backbone.layer1",
        "backbone.layer2",
        "backbone.layer3",
        "backbone.layer4",
        "backbone.fc",
    ]


def get_bibtex(model_identifier):
    return """@misc{he2015deepresiduallearningimage,
            title={Deep Residual Learning for Image Recognition}, 
            author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
            year={2015},
            eprint={1512.03385},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/1512.03385}, 
        }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
