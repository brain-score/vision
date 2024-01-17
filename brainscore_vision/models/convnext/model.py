import functools

import torch
import torchvision.models
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11976--11986},
  year={2022}
}"""

net_constructors = {
    "convnext_tiny": torchvision.models.convnext_tiny,
    "convnext_small": torchvision.models.convnext_small,
    "convnext_base": torchvision.models.convnext_base,
    "convnext_large": torchvision.models.convnext_large,
}

# these layer choices were not investigated in any depth, we blindly
# picked all high-level blocks
include_layer_names = {
    "features.0.0", "features.0.1",
    "features.1.0", "features.1.1", "features.1.2",
    "features.2.0", "features.2.1",
    "features.3.0", "features.3.1", "features.3.2",
    "features.4.0", "features.4.1",
    *{f"features.5.{i}" for i in range(27)},
    "features.6.0", "features.6.1", "features.7.0",
    "features.7.1", "features.7.2",
    "avgpool", "classifier",
}


def get_layers(net):
    assert net in net_constructors, f"Could not find ConvNeXt network: {net}"
    model = net_constructors[net](pretrained=True)
    return [layer for layer, _ in model.named_modules()
            if layer in include_layer_names]


def get_model(net):
    assert net in net_constructors, f"Could not find ConvNeXt network: {net}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = net_constructors[net](pretrained=True).to(device)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier=net,
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
