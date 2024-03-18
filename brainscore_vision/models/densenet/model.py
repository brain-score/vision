import functools

import torchvision.models
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}"""

net_constructors = {
    "densenet121": torchvision.models.densenet121,
    "densenet161": torchvision.models.densenet161,
    "densenet169": torchvision.models.densenet169,
    "densenet201": torchvision.models.densenet201,
}

# these layer choices were not investigated in any depth, we blindly picked all high-level blocks
include_layer_names = {
    'features.conv0', 'features.norm0', 'features.relu0', 'features.pool0', 'features.denseblock1',
    'features.transition1', 'features.denseblock2', 'features.transition2', 'features.denseblock3',
    'features.transition3', 'features.denseblock4', 'features.norm5', 'classifier',
}


def get_layers(net):
    assert net in net_constructors, f"Could not find DenseNet network: {net}"
    model = net_constructors[net](pretrained=True)
    return [layer for layer, _ in model.named_modules()
            if layer in include_layer_names]


def get_model(net):
    assert net in net_constructors, f"Could not find DenseNet network: {net}"
    model = net_constructors[net](pretrained=True)
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
