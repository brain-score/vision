"""
Resnet-50 BYOL model for Brain-Score submission.

Checkpoint from https://github.com/yaox12/BYOL-PyTorch.
"""

import functools
from collections import OrderedDict

import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


REPO_ID = "niels-c/resnet50_byol_linhead"
FILENAME = "resnet50_byol_linhead.safetensors"
REVISION = "05c4434c1f9e2fedf53a7ec2af7e1e94541205e5"


def load_model():
    """
    Load the network from its checkpoint on the Hugging Face Hub.

    Returns:
        nn.Module: A Sequential of "backbone", a ResNet-50 whose layer fc is the linear head (logits), in eval mode.
    """

    weights = load_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION))
    backbone = torchvision.models.resnet50(weights=None)
    backbone.load_state_dict(weights)
    return nn.Sequential(OrderedDict(backbone=backbone)).eval()


def get_model(name):
    assert name == "resnet50_byol_linhead"
    image_size = 224
    model = load_model()  # head at backbone.fc, logits
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier="resnet50_byol_linhead", model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == "resnet50_byol_linhead"
    return ["backbone.conv1", "backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4", "backbone.fc"]


def get_bibtex(model_identifier):
    return """@misc{grill2020bootstraplatentnewapproach,
      title={Bootstrap your own latent: A new approach to self-supervised Learning}, 
      author={Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
      year={2020},
      eprint={2006.07733},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.07733}, 
    }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
