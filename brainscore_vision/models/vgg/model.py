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
    "vgg11": torchvision.models.vgg11,
    "vgg11_bn": torchvision.models.vgg11_bn,
    "vgg13": torchvision.models.vgg13,
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16": torchvision.models.vgg16,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19": torchvision.models.vgg19,
    "vgg19_bn": torchvision.models.vgg19_bn,
}

include_layer_names = {
    'features.1', 'features.4', 'features.6', 'features.9', 'features.11', 'features.13',
    'features.15', 'features.18', 'features.20', 'features.22', 'features.24', 'features.27',
    'features.29', 'features.31', 'features.33', 'avgpool', 'classifier.1', 'classifier.4',
}


def get_layers(net):
    assert net in net_constructors, f"Could not find VGG network: {net}"
    model = net_constructors[net](pretrained=True)
    return [layer for layer, _ in model.named_modules()
            if layer in include_layer_names]


def get_model(net):
    assert net in net_constructors, f"Could not find VGG network: {net}"
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
