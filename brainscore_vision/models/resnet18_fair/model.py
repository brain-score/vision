import functools
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

# ResNet-18 trained on TinyImageNet-200.
# Architecture: standard ResNet-18 with conv1 stride changed from 2 to 1,
# preserving 64x64 spatial resolution after the stem (matches VOneNet layout).
#
# Weights location: bundled as model_weights.pth in this plugin directory.
# To host weights remotely instead, set WEIGHTS_URL to a public download link
# and remove model_weights.pth from the zip.

WEIGHTS_URL = None   # e.g. "https://huggingface.co/.../resolve/main/resnet18_fair.pth"


def get_model(name):
    assert name == 'resnet18_fair'

    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 200)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

    weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state_dict)
    elif WEIGHTS_URL is not None:
        import urllib.request, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            urllib.request.urlretrieve(WEIGHTS_URL, tmp.name)
            ckpt = torch.load(tmp.name, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            net.load_state_dict(state_dict)
            os.unlink(tmp.name)
    else:
        raise FileNotFoundError(
            'No weights found. Either place model_weights.pth next to model.py '
            'or set WEIGHTS_URL to a public download link.'
        )

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
