from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

import torchvision
import torch
import numpy as np
from PIL import Image
# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

MODEL_COMMITMENT = {
    "region2layer": {
        "V1": "layer2.0.relu",
        "V2": "layer2.0.relu",
        "V4": "layer2.0.relu",
        "IT": "layer4.0.relu",
    },
    "layers": ["layer2.0.relu", "layer4.0.relu"],
    "behavioral_readout_layer": "avgpool",
}

def val_transforms() -> "transforms.Compose":
    """Validation transformations."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def get_model(name):
    assert name == 'resnet18_barlortwins:standard_in1k_ba1024_ep100'
    model = torchvision.models.resnet18(pretrained=False)
    state_dict = torch.load("./resnet18.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    preprocessing = functools.partial(load_preprocess_images,transforms=val_transforms(), image_size=224)
    wrapper = PytorchWrapper(identifier='resnet18_barlortwins:standard_in1k_ba1024_ep100', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224

    brain_model = ModelCommitment(
        identifier=name,
        activations_model=wrapper,
        layers=MODEL_COMMITMENT["layers"],
        region_layer_map=MODEL_COMMITMENT["region2layer"],
        behavioral_readout_layer=MODEL_COMMITMENT["behavioral_readout_layer"],
    )
    return brain_model


def get_layers(name):
    assert name == 'resnet18_barlortwins:standard_in1k_ba1024_ep100'
    return ["layer2.0.relu", "layer4.0.relu"]


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
