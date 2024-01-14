import functools

import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

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

LAYERS = ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']


def get_model():
    model = torchvision.models.alexnet(pretrained=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet_untrained', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

if __name__ == '__main__':
    check_models.check_base_models(__name__)
