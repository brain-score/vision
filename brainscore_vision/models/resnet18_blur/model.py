import torch
import torchvision.models as models
import os
import functools
from torchvision import transforms
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

# Automatisch den Pfad zum aktuellen Skript ermitteln
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "r18_blur_only_best.pth")

# Transformationen für TinyImageNet (64x64)
def val_transforms():
    return transforms.Compose([
        transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

LAYER_NAMES = ["layer1", "layer3", "fc"]

def get_model(name):
    assert name == "resnet18_blur"
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 200)  # TinyImageNet: 200 Klassen
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    preprocessing = functools.partial(val_transforms())

    activations_model = PytorchWrapper(
        identifier="resnet18_blur",
        model=model,
        preprocessing=preprocessing,
        batch_size=8
    )
    activations_model.image_size = 64

    brain_model = ModelCommitment(
        identifier="resnet18_blur",
        activations_model=activations_model,
        layers=LAYER_NAMES
    )

    return brain_model

def get_layers(name):
    assert name == "resnet18_blur"
    return LAYER_NAMES