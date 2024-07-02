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
    return ["resnet50_eMMCR_eqp10_lm1V2"]


def get_model(name):
    assert name == "resnet50_eMMCR_eqp10_lm1V2"
    model = torchvision.models.resnet50(pretrained=False)
    url = "https://users.flatironinstitute.org/~tyerxa/equi_proj/training_checkpoints/classifiers/equi_proj_10_lmda_1/classifier.pt"
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="resnet50", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "resnet50_eMMCR_eqp10_lm1V2"
    layers = [
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer4.0",
        "layer4.1",
        "layer4.2",
        "avgpool",
        "fc",
    ]

    outs = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    return layers + outs


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
