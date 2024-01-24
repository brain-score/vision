import functools

import torch
import torchvision.models
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

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

net_constructors = {
    "resnet-18": torchvision.models.resnet18,
    "resnet-34": torchvision.models.resnet34,
    "resnet-50": torchvision.models.resnet50,
    "resnet-101": torchvision.models.resnet101,
    "resnet-152": torchvision.models.resnet152,
}

net_units = {
    "resnet-50": [3, 4, 6, 3],
    "resnet-101": [3, 4, 23, 3],
}


def get_layers(net):
    assert net in net_constructors, f"Could not find ResNet network: {net}"
    if net in net_units:
        units = net_units[net]
        layers = ['conv1']
        for v in {1, 2}:
            layers += [f"layer{block + 1}.{unit}.conv{v}"
                       for block, block_units in enumerate(units) for unit in range(block_units)]
        return layers
    model = net_constructors[net](pretrained=True)
    return [layer for layer, _ in model.named_modules()][1:]


def get_model(net):
    assert net in net_constructors, f"Could not find ResNet network: {net}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
