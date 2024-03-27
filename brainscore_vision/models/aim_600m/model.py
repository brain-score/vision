import functools
import os
from pathlib import Path

import torchvision
import torch
import numpy as np
from PIL import Image

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers import download_weights
from brainscore_vision.model_helpers.check_submission import check_models


# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

from aim.utils import load_pretrained
from aim.torch.data import val_transforms


os.environ["RESULTCACHING_DISABLE"] = "1"

layers = [
    "trunk.blocks.0.mlp.act",
    "trunk.blocks.1.mlp.act",
    "trunk.blocks.10.mlp.act",
    "trunk.blocks.11.mlp.act",
    "trunk.blocks.12.mlp.act",
    "trunk.blocks.13.mlp.act",
    "trunk.blocks.14.mlp.act",
    "trunk.blocks.15.mlp.act",
    "trunk.blocks.16.mlp.act",
    "trunk.blocks.17.mlp.act",
    "trunk.blocks.18.mlp.act",
    "trunk.blocks.19.mlp.act",
    "trunk.blocks.2.mlp.act",
    "trunk.blocks.20.mlp.act",
    "trunk.blocks.21.mlp.act",
    "trunk.blocks.22.mlp.act",
    "trunk.blocks.23.mlp.act",
    "trunk.blocks.3.mlp.act",
    "trunk.blocks.4.mlp.act",
    "trunk.blocks.5.mlp.act",
    "trunk.blocks.6.mlp.act",
    "trunk.blocks.7.mlp.act",
    "trunk.blocks.8.mlp.act",
    "trunk.blocks.9.mlp.act",
    "trunk.post_trunk_norm",
    "trunk.post_transformer_layer",
]


def get_model_list():
    return ["aim_600m"]


def custom_image_preprocess(images, transforms):
    # print(images, transforms)
    images = [Image.open(image).convert('RGB') for image in images]
    images = [transforms(image) for image in images]
    images = [np.array(image) for image in images]
    images = np.stack(images)

    return images


def get_model(name):
    assert name == "aim_600m"
    # model = torch.hub.load("apple/ml-aim", "aim_600M")
    model = load_pretrained("aim-600M-2B-imgs", backend="torch")
    preprocessing = functools.partial(
        custom_image_preprocess, transforms=val_transforms()
    )
    activations_model = PytorchWrapper(
        identifier=name, model=model, preprocessing=preprocessing, batch_size=8
    )
    activations_model.image_size = 224

    return activations_model


def get_layers(name):
    assert name == "aim_600m"
    return layers


def get_bibtex(model_identifier):
    return """@misc{elnouby2024scalable,
      title={Scalable Pre-training of Large Autoregressive Image Models},
      author={Alaaeldin El-Nouby and Michal Klein and Shuangfei Zhai and Miguel Angel Bautista and Alexander Toshev and Vaishaal Shankar and Joshua M Susskind and Armand Joulin},
      year={2024},
      eprint={2401.08541},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
        }"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
