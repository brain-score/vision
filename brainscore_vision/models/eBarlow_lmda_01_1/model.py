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
from collections import OrderedDict

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models


def get_model_list():
    return ["eBarlow_lmda_01_1"]


def get_model(name):
    assert name == "eBarlow_lmda_01_1"
    url = "https://users.flatironinstitute.org/~tyerxa/equi_proj/training_checkpoints/fresh/paired/lmda_0.1/Barlow_1/latest-rank0"
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))["state"]["model"]
    model = load_composer_classifier(state_dict)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def load_composer_classifier(sd):
    model = torchvision.models.resnet.resnet50()
    new_sd = OrderedDict()
    for k, v in sd.items():
        if 'lin_cls' in k:
            new_sd['fc.' + k.split('.')[-1]] = v
        if ".f." not in k:
            continue
        parts = k.split(".")
        idx = parts.index("f")
        new_k = ".".join(parts[idx + 1 :])
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=True)
    return model

def get_layers(name):
    assert name == "eBarlow_lmda_01_1"

    outs = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    outs = ["layer4"]
    return outs


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
