import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models


def get_model_list():
    return ['alexnet']


def get_model(name):
    assert name == 'alexnet'
    model = torchvision.models.alexnet(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'alexnet'
    return ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
            'classifier.2', 'classifier.5']


def get_bibtex(model_identifier):
    return """@incollection{NIPS2012_4824,
                title = {ImageNet Classification with Deep Convolutional Neural Networks},
                author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                booktitle = {Advances in Neural Information Processing Systems 25},
                editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
                pages = {1097--1105},
                year = {2012},
                publisher = {Curran Associates, Inc.},
                url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
