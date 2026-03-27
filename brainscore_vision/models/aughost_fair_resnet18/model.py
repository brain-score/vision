from brainscore_vision.model_helpers.check_submission import check_models
import functools
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

"""
AugGhost Fair ResNet18
======================
This is a MODIFIED ResNet18 architecture, NOT standard ResNet18!

Key architecture difference:
- conv1 has stride=1 (instead of standard stride=2)
- This preserves higher spatial resolution (32x32 vs 16x16 after stem)
- This "Fair" modification matches VOneNet's resolution for fair comparison

Training details:
- Dataset: TinyImageNet-200 (200 classes, 64x64 images)
- Augmentation: RandomGhosting (max_shift=2, alpha=0.5, p=0.5)
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
"""

WEIGHTS_URL = None  # TODO: Host weights and add URL here for website submission


def get_model(name):
    assert name == 'aughost_fair_resnet18'
    
    # Create ResNet18 with TinyImageNet-200 classes
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    
    # ========================================
    # CRITICAL: FAIR RESNET18 MODIFICATION
    # Change stride from 2 to 1 to preserve spatial resolution
    # This matches VOneNet's 32x32 output after stem (vs standard 16x16)
    # ========================================
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    # Load weights
    weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {weights_path}")
    elif WEIGHTS_URL is not None:
        import urllib.request
        import tempfile
        print(f"Downloading weights from {WEIGHTS_URL}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            urllib.request.urlretrieve(WEIGHTS_URL, tmp.name)
            state_dict = torch.load(tmp.name, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            os.unlink(tmp.name)
        print("Weights loaded successfully")
    else:
        print("WARNING: No weights found - using random initialization")
    
    model.eval()
    
    # Use 64x64 input size (native TinyImageNet resolution)
    # Brain-Score will resize stimuli accordingly
    preprocessing = functools.partial(load_preprocess_images, image_size=64)
    wrapper = PytorchWrapper(identifier='aughost_fair_resnet18', model=model, preprocessing=preprocessing)
    wrapper.image_size = 64
    return wrapper


def get_layers(name):
    assert name == 'aughost_fair_resnet18'
    # Fair ResNet18 layers (same as standard, but conv1 has stride=1)
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    return """
@misc{aughost_fair_resnet18_2024,
    title={AugGhost Fair ResNet18: Motion-Robust Training with Spatial Resolution Preservation},
    author={User},
    year={2024},
    note={ResNet18 with stride-1 conv1 (Fair comparison to VOneNet), trained on TinyImageNet-200 with RandomGhosting augmentation (max_shift=2, alpha=0.5, p=0.5)}
}
"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
