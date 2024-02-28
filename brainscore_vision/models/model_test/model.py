
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import torch.nn as nn
import os
import numpy as np
from pathlib import Path
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from torchvision.models import resnet50


# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


def get_model_list():
    return ['yudixie_resnet50_imagenet1kpret_0_240222']


def get_model(name):
    assert name == 'yudixie_resnet50_imagenet1kpret_0_240222'

    pytorch_device = torch.device('cpu')

    weigth_url = f'https://yudi-brainscore-models.s3.amazonaws.com/{name}.pth'
    os.system(f'wget "{weigth_url}"')
    load_path = Path(__file__).parent.joinpath(f'{name}.pth')

    pytorch_model = resnet50()
    pytorch_model.fc = nn.Linear(pytorch_model.fc.in_features, 674)
    pytorch_model = pytorch_model.to(pytorch_device)

    # load model from saved weights
    saved_state_dict = torch.load(load_path, map_location=pytorch_device)
    state_dict = {}
    for k, v in saved_state_dict.items():
        if k.startswith('_orig_mod.'):
            # for compiled models
            state_dict[k[10:]] = v
        else:
            state_dict[k] = v
    pytorch_model.load_state_dict(state_dict, strict=True)
    print(f'Loaded model from {load_path}')

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name,
                             model=pytorch_model,
                             preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'yudixie_resnet50_imagenet1kpret_0_240222'
    return ['conv1','layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
    # get_model('yudixie_resnet50_imagenet1kpret_0_240222')
