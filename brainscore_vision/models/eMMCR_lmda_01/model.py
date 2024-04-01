from brainscore_vision.model_helpers.check_submission import check_models
import functools
import os
from urllib.request import urlretrieve
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
import torch

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models


def get_model_list():
    return ["eMMCR_lmda_01"]


def get_model(name):
    assert name == "eMMCR_lmda_01"
    model = torchvision.models.resnet50(pretrained=False)
    url = "https://users.flatironinstitute.org/~tyerxa/equi_proj/training_checkpoints/classifiers/mmcr/equi_matched_0.1/classifier.pt"
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "eMMCR_lmda_01"

    outs = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    return outs


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
