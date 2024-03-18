import functools

import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@inproceedings{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={10096--10106},
  year={2021},
  organization={PMLR}
}"""

net_constructors = {
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
    "efficientnet_v2_l": torchvision.models.efficientnet_v2_l,
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
    model = net_constructors[net](pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier=net,
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
