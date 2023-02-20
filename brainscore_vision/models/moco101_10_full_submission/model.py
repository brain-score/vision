import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import torch
import pathlib
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models


def get_model_list():
    return ['moco_101_10']


def get_model(name):
    cur_path=pathlib.Path(__file__).parent.resolve()
    assert name == 'moco_101_10'
    # # model = torchvision.models.alexnet(pretrained=True)
    # state_dict = torch.load(f'{cur_path}/moco_30.pth.tar',map_location=torch.device('cpu'))['state_dict']
    # resnet = torchvision.models.resnet101(pretrained=False)
    # for k in list(state_dict.keys()):
    #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') :
    #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #     del state_dict[k]
    # msg = resnet.load_state_dict(state_dict, strict=False)
    resnet=torch.load(f'{cur_path}/moco_10_encoder.pth.tar')
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper( model=resnet, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'moco_101_10'
    # return  ['layer3.0','layer3.1','layer3.2','layer3.3']
    return  ['layer1.0','layer1.1','layer1.2','layer2.0','layer2.1','layer2.2','layer2.3','layer3.0','layer3.1','layer3.2','layer3.3','layer3.4','layer3.5','layer3.6','layer3.7','layer3.8','layer3.9','layer3.10','layer3.11','layer3.12','layer3.13','layer3.14','layer3.15','layer3.16','layer3.17','layer3.18','layer3.19','layer3.20','layer3.21','layer3.22','layer4.0','layer4.1','layer4.2']


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
