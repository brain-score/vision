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
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


os.environ["RESULTCACHING_DISABLE"] = "1"

MODEL_NAME = "resnet18_imagenet21kP"

MODEL_COMMITMENT = {
    "region2layer": {
        "V1": "layer2.0.relu",
        "V2": "layer2.0.relu",
        "V4": "layer2.0.relu",
        "IT": "layer4.0.relu"
    },
    "layers": [
        "layer2.0.relu",
        "layer4.0.relu"
    ],
    "behavioral_readout_layer": "avgpool"
}

def val_transforms() -> "transforms.Compose":
    """Validation transformations."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
                ),
        ]
    )


def get_model_list():
    return [MODEL_NAME]


def custom_image_preprocess(images, transforms):
    # print(images, transforms)
    images = [Image.open(image).convert('RGB') for image in images]
    images = [transforms(image) for image in images]
    images = [np.array(image) for image in images]
    images = np.stack(images)

    return images


def get_model(name):
    assert name == MODEL_NAME
    # model = torch.hub.load("apple/ml-aim", "aim_600M")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 10450)
    # state_dict = torch.load("checkpoint.pt", map_location="cpu")
    state_dict = torch.utils.model_zoo.load_url("https://adf349cdkj2349.blob.core.windows.net/model-checkpoints/resnet18_imagenet21kP.pt", map_location="cpu")
    weights = state_dict["state"]["model"]
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    
    preprocessing = functools.partial(
        custom_image_preprocess, transforms=val_transforms()
    )
    activations_model = PytorchWrapper(
        identifier=name, model=model, preprocessing=preprocessing, batch_size=8
    )
    activations_model.image_size = 224
    
    brain_model = ModelCommitment(
        identifier = MODEL_NAME, 
        activations_model = activations_model, 
        layers = MODEL_COMMITMENT['layers'],
        region_layer_map = MODEL_COMMITMENT['region2layer'],
        behavioral_readout_layer = MODEL_COMMITMENT['behavioral_readout_layer']
    )

    return brain_model


def get_commitment(name):
    assert name == MODEL_NAME
    return MODEL_COMMITMENT


def get_bibtex(model_identifier):
    return """@INPROCEEDINGS{7780459,
    author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
    booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    title={Deep Residual Learning for Image Recognition}, 
    year={2016},
    volume={},
    number={},
    utl={https://doi.org/10.1109/CVPR.2016.90},
    pages={770-778},
    doi={10.1109/CVPR.2016.90}}
    """


if __name__ == "__main__":
    check_models.check_brain_models(__name__)
