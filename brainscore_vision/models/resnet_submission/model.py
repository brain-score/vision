# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

# from brainscore import score_model
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

# from brainscore import score_model
import torchvision.models as models
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

BIBTEX = ""
LAYERS = [
    "conv1",
    "layer1.1.conv2",
    "layer2.1.conv2",
    "layer3.1.conv2",
    "layer4.1.conv2",
]


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model():
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)  # for cifar10 training
    download_weights(
        bucket="brainscore-vision",
        folder_path="models/resnet_submission",
        filename_version_sha=[
            (
                "best_resnet18.pth",
                "DPvfIHTBCBISvGcRpZ4Rca21.gmnOR0g",
                "8330c55ea446504d500ab5847434606b4d76d9e0",
            )
        ],
        save_directory=Path(__file__).parent,
    )
    ckpt_path = Path(__file__).parent / "best_resnet18.pth"
    model.load_state_dict(
        torch.load(ckpt_path)
    )
    model.fc = nn.Linear(512, 1000)  # imagenet output mapping
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="custom_resnet_submission", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
