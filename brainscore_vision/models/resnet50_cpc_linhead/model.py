"""
Resnet-50 CPC model for Brain-Score submission.

Checkpoint from https://huggingface.co/niels-c/resnet50_cpc_linhead.
"""

import functools
from collections import OrderedDict

import timm
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)
from brainscore_vision.model_helpers.check_submission import check_models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

REPO_ID = "niels-c/resnet50_cpc_linhead"
FILENAME = "resnet50_cpc_linhead.safetensors"


def load_model():
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", a ResNet-50 whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

    backbone = timm.create_model("resnet50", num_classes=0)
    backbone.feature_dim = backbone.head_hidden_size
    in_forward = isinstance(backbone, nn.Sequential) or isinstance(
        getattr(backbone, "fc", None), nn.Identity
    )
    backbone.fc = nn.Linear(backbone.feature_dim, 1000)
    if not in_forward:
        backbone.register_forward_hook(
            lambda module, inputs, features: module.fc(features)
        )
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


def get_model(name):
    assert name == "resnet50_cpc_linhead"
    image_size = 224
    model = load_model()  # head at backbone.fc, logits
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(
        identifier="resnet50_cpc_linhead", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == "resnet50_cpc_linhead"
    return [
        "backbone.conv1",
        "backbone.layer1",
        "backbone.layer2",
        "backbone.layer3",
        "backbone.layer4",
        "backbone.fc",
    ]


def get_bibtex(model_identifier):
    return """@misc{oord2019representationlearningcontrastivepredictive,
                title={Representation Learning with Contrastive Predictive Coding}, 
                author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
                year={2019},
                eprint={1807.03748},
                archivePrefix={arXiv},
                primaryClass={cs.LG},
                url={https://arxiv.org/abs/1807.03748}, 
            }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
