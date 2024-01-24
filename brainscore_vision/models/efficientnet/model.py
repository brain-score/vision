import functools

import torch
import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}"""

net_constructors = {
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    "efficientnet_b1": torchvision.models.efficientnet_b1,
    "efficientnet_b2": torchvision.models.efficientnet_b2,
    "efficientnet_b3": torchvision.models.efficientnet_b3,
    "efficientnet_b4": torchvision.models.efficientnet_b4,
    "efficientnet_b5": torchvision.models.efficientnet_b5,
    "efficientnet_b6": torchvision.models.efficientnet_b6,
    "efficientnet_b7": torchvision.models.efficientnet_b7,
}

include_layer_names = {
    'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6',
    'features.7', 'features.8', 'avgpool', 'classifier.0', 'classifier.1',
}


def get_layers(net):
    assert net in net_constructors, f"Could not find EfficientNet network: {net}"
    model = net_constructors[net](pretrained=True)
    return [layer for layer, _ in model.named_modules()
            if layer in include_layer_names]


def get_model(net):
    assert net in net_constructors, f"Could not find EfficientNet network: {net}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net_constructors[net](pretrained=True).to(device)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier=net,
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
