import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
#
# from test import test_models


def get_model_list():
    return ['alexnet','googlenet','resnet18']


def get_model(name):
    if name=='alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='alexnet', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='googlenet', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='squeezenet1_0', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'squeezenet1_1':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='squeezenet1_1', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='resnet18', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='vgg16', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    elif name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier='vgg19', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper


def get_layers(name):
    if name == 'alexnet':
        return ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
                'classifier.2', 'classifier.5']
    elif name == 'googlenet':
        return ['maxpool1', 'maxpool2', 'maxpool3', 'maxpool4', 'avgpool']
    elif name == 'squeezenet1_0':
        return ['features.2','features.6', 'features.11', 'classifier.2']
    elif name == 'squeezenet1_1':
        return ['features.2','features.6', 'features.11', 'classifier.2']
    elif name == 'resnet18':
        return ['maxpool','avgpool']
    elif name == 'vgg16':
        return ['features.4','features.9','features.16','features.23','features.30','classifier.1','classifier.4']
    elif name == 'vgg19':
        return ['features.4','features.9','features.18','features.27','features.36','classifier.1','classifier.4']


if __name__ == '__main__':
    test_models.test_base_models(__name__)
