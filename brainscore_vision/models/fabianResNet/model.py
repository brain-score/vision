from brainscore_vision.model_helpers.check_submission import check_models
import functools
from torchvision.models import resnet18
from torch import nn
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


class FabianResNet(torch.nn.Module):
    def __init__(self):
        super(FabianResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout to avoid overfitting
            nn.Linear(self.model.fc.in_features, 200)  # Tiny ImageNet has 200 classes
        )

    def forward(self, x):
        activations = {}

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        activations["conv1"] = x

        x = self.model.layer1(x)
        activations["layer1"] = x
        x = self.model.layer2(x)
        activations["layer2"] = x
        x = self.model.layer3(x)
        activations["layer3"] = x
        x = self.model.layer4(x)
        activations["layer4"] = x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        activations["fc"] = x

        return activations

def get_model_list():
    return ['fabianResNet']


def get_model(name):
    assert name == 'fabianResNet'
    preprocessing = functools.partial(load_preprocess_images, image_size=64)
    activations_model = PytorchWrapper(identifier='fabianResNet', model=FabianResNet(), preprocessing=preprocessing)
    model = ModelCommitment(identifier='fabianResNet', activations_model=activations_model,
                            layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'])
    model.image_size = 64
    return model


def get_layers(name):
    assert name == 'fabianResNet'
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    return """FabianResNetModel"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
