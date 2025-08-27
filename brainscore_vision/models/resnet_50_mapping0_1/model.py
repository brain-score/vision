import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .evnet.evnet import EVNet
import torch
import torchvision
import os
import wget

def get_model(name):
    assert name == 'resnet_50_mapping0_1'
    weight_file = '.weights.pth'
    bucket = 'evnets-model-weights'
    rel_path = 'resnet_50_0.pth'
    wget.download(
        f'https://{bucket}.s3.eu-west-2.amazonaws.com/models/{rel_path}',
        out=weight_file
        )
    model = EVNet(
        with_retinablock=False, with_voneblock=False, model_arch='resnet50',
        image_size=224, visual_degrees=7, num_classes=1000
        )
    model.to(torch.device('cpu'))
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'), weights_only=False)
    os.remove(weight_file)
    model.load_state_dict(checkpoint['model'])
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(.5,.5,.5),
        normalize_std=(.5,.5,.5)
        )
    wrapper = PytorchWrapper(
        identifier='resnet_50_mapping0_1',
        model=model, preprocessing=preprocessing
        )
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_mapping0_1'
    return ['model.conv1', 'model.bn1', 'model.relu', 'model.maxpool',
        'model.layer1.0', 'model.layer1.1', 'model.layer1.2',
        'model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3',
        'model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
        'model.layer3.4', 'model.layer3.5',
        'model.layer4.0', 'model.layer4.1', 'model.layer4.2',
        'model.avgpool']


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
