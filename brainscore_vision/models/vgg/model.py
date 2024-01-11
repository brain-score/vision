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

def get_layers(net):
    if net == "vgg-16":
        model = torchvision.models.vgg16(pretrained=True)
    elif net == "vgg-19":
        model = torchvision.models.vgg19(pretrained=True)
    else:
        raise NotImplementedError()
    return [layer for layer, _ in model.named_modules()]

def get_model(net):
    if net == "vgg-16":
        model = torchvision.models.vgg16(pretrained=True)
    elif net == "vgg-19":
        model = torchvision.models.vgg19(pretrained=True)
    else:
        raise NotImplementedError()
    
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet-18', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
