from brainscore_vision.model_helpers.check_submission import check_models
import functools
import os
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
import torch

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models


def get_model_list():
    return ['dbp_resnet50_julios']


def get_model(name):
    assert name == 'dbp_resnet50_julios'
    ckpt_name = 'dbp_export.pt'
    download_weights(
        bucket='brainscore-vision', 
        folder_path='models/dbp_resnet50_julios',
        filename_version_sha=[(ckpt_name, 'HpyXY.W.WtS8zD.wJ7BVfvSbQDX.1Md4', '6efa18bfe4b70221d7e88ac2d2279d08042f23da')],
        save_directory=Path(__file__).parent
    )
    ckpt_path = Path(__file__).parent / ckpt_name
    model = torch.load(ckpt_path)    
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    print(preprocessing)
    wrapper = PytorchWrapper(identifier='dbp_resnet50_julios', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'dbp_resnet50_julios'
    return ['layer1', 'layer2', 'layer3', 'layer4']  


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
