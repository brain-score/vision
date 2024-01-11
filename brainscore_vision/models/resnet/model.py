import functools

import torchvision.models

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

LAYERS = {
    'resnet-18':
        ['conv1'] +
        ['layer1.0.relu', 'layer1.1.relu'] +
        ['layer2.0.relu', 'layer2.0.downsample.0', 'layer2.1.relu'] +
        ['layer3.0.relu', 'layer3.0.downsample.0', 'layer3.1.relu'] +
        ['layer4.0.relu', 'layer4.0.downsample.0', 'layer4.1.relu'] +
        ['avgpool'],
    'resnet-34':
        ['conv1'] +
        ['layer1.0.conv2', 'layer1.1.conv2', 'layer1.2.conv2'] +
        ['layer2.0.downsample.0', 'layer2.1.conv2', 'layer2.2.conv2', 'layer2.3.conv2'] +
        ['layer3.0.downsample.0', 'layer3.1.conv2', 'layer3.2.conv2', 'layer3.3.conv2',
         'layer3.4.conv2', 'layer3.5.conv2'] +
        ['layer4.0.downsample.0', 'layer4.1.conv2', 'layer4.2.conv2'] +
        ['avgpool'],
}


def get_model(net):
    if net == "resnet-18":
        model = torchvision.models.resnet18(pretrained=True)
    elif net == "resnet-34":
        model = torchvision.models.resnet34(pretrained=True)
    else:
        raise NotImplementedError()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="net", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper
