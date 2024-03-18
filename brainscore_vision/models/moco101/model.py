import functools
import pathlib
from pathlib import Path

import torch

from brainscore_vision.model_helpers import download_weights
from brainscore_vision.model_helpers.activations.pytorch import (
    PytorchWrapper,
    load_preprocess_images,
)

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py


BIBTEX = """@incollection{NIPS2012_4824,
    title = {ImageNet Classification with Deep Convolutional Neural Networks},
    author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
    booktitle = {Advances in Neural Information Processing Systems 25},
    editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
    pages = {1097--1105},
    year = {2012},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
}"""

LAYERS = [
    "layer1.0", "layer1.1", "layer1.2", "layer2.0", "layer2.1", "layer2.2", "layer2.3",
    "layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5", "layer3.6",
    "layer3.7", "layer3.8", "layer3.9", "layer3.10", "layer3.11", "layer3.12", "layer3.13",
    "layer3.14", "layer3.15", "layer3.16", "layer3.17", "layer3.18", "layer3.19", "layer3.20",
    "layer3.21", "layer3.22", "layer4.0", "layer4.1", "layer4.2",
]

models_folder_filename_version_sha = {
    "moco_101_20": (
        "models/moco101_20_full_submission",
        "moco_20_encoder.pth.tar",
        "ZlaH17VoVJ1xpAo6szH_iwLuVtSwBRsW",
        "d1fcdd77ec80d71376c50646ba854aeccf223ec1",
    ),
    "moco_101_30": (
        "models/moco101_30_full_submission",
        "moco_30_encoder.pth.tar",
        "t26uKPlpI05UxZFlHc5.doMyJ2oIS3pv",
        "363c88834119a2b21b31f6492fe83762790985f2",
    ),
    "moco_101_40": (
        "models/moco101_40_full_submission",
        "moco_40_encoder.pth.tar",
        "Tu37bJzXX35njS7ivKeFxs8_MU6NDIVm",
        "165637605d0817c4047260aa55601406f14f8a2f",
    ),
    "moco_101_50": (
        "models/moco101_50_full_submission",
        "moco_50_encoder.pth.tar",
        "6Ez_9iSFNlLVTCZLnj.EYxOaOPj83YzE",
        "22846b0aeebec736e9463bdc39e015af5449eb71",
    ),
    "moco_101_60": (
        "models/moco101_60_full_submission",
        "moco_60_encoder.pth.tar",
        "V59i4oQtW8AOqW9TQkV9GevAsEZV4t4Y",
        "1897dd4e4cb1afe5d731787b92022c1010471dd0",
    ),
    "moco_101_70": (
        "models/moco101_70_full_submission",
        "moco_70_encoder.pth.tar",
        "ojekGjeQ4Y9IMq4iGxvPnIhbIK53rr1u",
        "fcdf0e24350c2666b49765cd0daa23544d2e739e",
    ),
}


def get_model(name):
    assert name in models_folder_filename_version_sha
    
    cur_path = pathlib.Path(__file__).parent.resolve()
    folder_path, filename, version, sha = models_folder_filename_version_sha[name]
    download_weights(
        bucket="brainscore-vision",
        folder_path=folder_path,
        filename_version_sha=[(
            filename,
            version,
            sha,
        )],
        save_directory=Path(__file__).parent,
    )
    resnet = torch.load(f"{cur_path}/{filename}")
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(model=resnet, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


if __name__ == "__main__":
    check_models.check_base_models(__name__)
