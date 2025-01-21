import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
import torch
from .evnet.evnet import EVNet

def get_model(name):
    assert name == 'resnet_50_2'
    model = EVNet(
        with_retinablock=False, with_voneblock=False,
        model_arch='resnet50', image_size=224, num_classes=1000
        )
    weight_file = load_weight_file(
        bucket="evnets-model-weights",
        relative_path="resnet_50_1.pth",
        sha1="aad77aaf2213858ef49c36b97d7b52855dc6e168",
        version_id="null"
        )
    model.to(torch.device('cpu'))
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    preprocessing = functools.partial(
        load_preprocess_images,
        image_size=224,
        normalize_mean=(.5,.5,.5),
        normalize_std=(.5,.5,.5)
        )
    wrapper = PytorchWrapper(
        identifier='resnet_50_2',
        model=model, preprocessing=preprocessing
        )
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'resnet_50_2'
    return ['model.conv1','model.layer1', 'model.layer2', 'model.layer3', 'model.layer4', 'model.fc']

def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
