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
    return ["CE-Barlow_lmda_03"]


def get_model(name):
    assert name == "CE-Barlow_lmda_03"
    lmda_map = {
        '00': 0.0,
        '0001': 0.001,
        '01': 0.1,
        '02': 0.2,
        '03': 0.3,
        '04': 0.4,
        '05': 0.5,
    }
    objective_name = name.split("_")[0].split("-")[1]
    lmda_str = name.split("_")[-1]
    lmda = lmda_map[lmda_str]
    ckpt_location = f'~tyerxa/equi_proj/training_checkpoints/final/resnet_50/{objective_name}/lmda_{lmda}/ep100-ba62500-rank0'
    url = f'https://users.flatironinstitute.org/{ckpt_location}'
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"), weights_only=False)["state"]["model"]
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
    assert name == "CE-Barlow_lmda_03"

    outs = ["layer4"]
    return outs


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
