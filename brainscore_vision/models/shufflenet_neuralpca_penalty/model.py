import os
import torch
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0
from urllib.request import urlretrieve
import gdown
from torch import Tensor

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

def load_model():
    model = shufflenet_v2_x1_0().to('cpu')
    
    url = "https://drive.google.com/file/d/1k-ccTqrVKi_xsTjDkZuRiUGqrFvUXbe-/view?usp=sharing"
    output = "shufflenet_penalized.pt"
    gdown.download(url, output,fuzzy=True)
    state = torch.load(output, map_location=torch.device('cpu'))
    state_dict = state['state_dict']
    os.remove(output)
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.shufflenet.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict,strict=False)
    return model

def get_model(name):
    assert name == 'shufflenet_neuralpca_penalty'
    model = load_model()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='shufflenet_neuralpca_penalty', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'shufflenet_neuralpca_penalty'
    return ['conv1', 'stage2', 'stage3', 'stage4', 'conv5', 'fc']

def get_bibtex(model_identifier):
    return """
    @misc{shufflenet_neuralpca_penalty,
    title={Shufflenet with neural predictivity penalty},
    author={William Qian},
    year={2025},
    }
    """

if __name__ == '__main__':
    check_models.check_base_models(__name__)
