import torch
from pathlib import Path
from torchvision.models import resnet18
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from collections import OrderedDict
from urllib.request import urlretrieve
import functools

# Custom model loader
def get_model(name):
    assert name == 'barlow_twins_custom'

    url = " https://www.dropbox.com/scl/fi/db5yp3hols5sucujanimx/barlow_twins_weights.pth?rlkey=nalge9jixfeqorazwu4xqdbd8&st=yqf3qkaj&dl=1"
    fh = urlretrieve(url)
    state_dict = torch.load(fh[0], map_location=torch.device("cpu"))
    model = resnet18(pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    print(model)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    activations_model = PytorchWrapper(identifier='barlow_twins_custom', model=model, preprocessing=preprocessing)

    
    return ModelCommitment(
        identifier='barlow_twins_custom',
        activations_model=activations_model,
        layers=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    )

def get_model_list():
    return ['barlow_twins_custom']

# Specify layers to test
def get_layers(name):
    assert name == 'barlow_twins_custom'
    return ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


if __name__ == '__main__':

    check_models.check_base_models(__name__)
