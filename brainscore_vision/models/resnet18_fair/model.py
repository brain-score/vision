import functools
import torch
import torch.nn as nn
from torchvision.models import resnet18
from brainscore_core.supported_data_standards.brainio.s3 import load_weight_file
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

# ResNet-18 trained on TinyImageNet-200.
# Architecture: standard ResNet-18 with conv1 stride changed from 2 to 1,
# preserving 64x64 spatial resolution after the stem (matches VOneNet layout).


def get_model(name):
    assert name == 'resnet18_fair'

    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 200)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

    weights_path = load_weight_file(
        bucket="brainscore-storage",
        folder_name="brainscore-vision/models",
        relative_path="resnet18_fair/model_weights.pth",
        version_id="null",
        sha1="04b939f367b2044a3e1dff55e861330c8ff7ef52",
    )
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    net.load_state_dict(state_dict)

    net.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier='resnet18_fair',
        model=net,
        preprocessing=preprocessing,
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet18_fair'
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    return ""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
