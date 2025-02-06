import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torchvision
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of resnet50.
The model template can be found at the following URL:
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
'''

MODEL = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')  # use V2 weights


def get_model(name):
    assert name == 'resnet_50_v2'
    preprocessing = functools.partial(load_preprocess_images, image_size=224, preprocess_type='inception')
    wrapper = PytorchWrapper(identifier=name, model=MODEL, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet_50_v2'
    layer_names = (['conv1'] + [f'layer1.{i}' for i in range(3)] +
                   [f'layer2.{i}' for i in range(4)] +
                   [f'layer3.{i}' for i in range(6)] +
                   [f'layer4.{i}' for i in range(3)] + ['avgpool'])
    return layer_names
