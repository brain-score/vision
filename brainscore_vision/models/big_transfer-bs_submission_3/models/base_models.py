import io
import sys
import torch
import requests
import functools
import numpy as np
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models

sys.path.append('../')
import bit_pytorch.models as models


BIT_HOME_URL = 'https://storage.googleapis.com/bit_models/'
ALL_MODELS = list(models.KNOWN_MODELS.keys())
R50_LAYERS = [
    'body.block1.unit01.relu', 'body.block1.unit02.relu',
    'body.block1.unit03.relu', 'body.block2.unit01.relu',
    'body.block2.unit02.relu', 'body.block2.unit03.relu',    
    'body.block2.unit04.relu', 'body.block3.unit01.relu',
    'body.block3.unit02.relu', 'body.block3.unit03.relu',
    'body.block3.unit04.relu', 'body.block3.unit05.relu',
    'body.block3.unit06.relu', 'body.block4.unit01.relu',
    'body.block4.unit02.relu', 'body.block4.unit03.relu'
]
R101_LAYERS = [
    'body.block1.unit01.relu', 'body.block1.unit02.relu',
    'body.block1.unit03.relu', 'body.block2.unit01.relu',    
    'body.block2.unit02.relu', 'body.block2.unit03.relu',
    'body.block2.unit04.relu', 'body.block3.unit01.relu',
    'body.block3.unit02.relu', 'body.block3.unit03.relu',
    'body.block3.unit04.relu', 'body.block3.unit05.relu',
    'body.block3.unit06.relu', 'body.block3.unit07.relu',
    'body.block3.unit08.relu', 'body.block3.unit09.relu',
    'body.block3.unit10.relu', 'body.block3.unit11.relu',
    'body.block3.unit12.relu', 'body.block3.unit13.relu',
    'body.block3.unit14.relu', 'body.block3.unit15.relu',
    'body.block3.unit16.relu', 'body.block3.unit17.relu',
    'body.block3.unit18.relu', 'body.block3.unit19.relu',
    'body.block3.unit20.relu', 'body.block3.unit21.relu',
    'body.block3.unit22.relu', 'body.block3.unit23.relu',
    'body.block4.unit01.relu', 'body.block4.unit02.relu',
    'body.block4.unit03.relu'
]
R152_LAYERS = [
    'body.block1.unit01.relu', 'body.block1.unit02.relu',
    'body.block1.unit03.relu', 'body.block2.unit01.relu',
    'body.block2.unit02.relu', 'body.block2.unit03.relu',
    'body.block2.unit04.relu', 'body.block2.unit05.relu',
    'body.block2.unit06.relu', 'body.block2.unit07.relu',
    'body.block2.unit08.relu', 'body.block3.unit01.relu',
    'body.block3.unit02.relu', 'body.block3.unit03.relu',
    'body.block3.unit04.relu', 'body.block3.unit05.relu',
    'body.block3.unit06.relu', 'body.block3.unit07.relu',
    'body.block3.unit08.relu', 'body.block3.unit09.relu',
    'body.block3.unit10.relu', 'body.block3.unit11.relu',
    'body.block3.unit12.relu', 'body.block3.unit13.relu',
    'body.block3.unit14.relu', 'body.block3.unit15.relu',
    'body.block3.unit16.relu', 'body.block3.unit17.relu',
    'body.block3.unit18.relu', 'body.block3.unit19.relu',
    'body.block3.unit20.relu', 'body.block3.unit21.relu',
    'body.block3.unit22.relu', 'body.block3.unit23.relu',
    'body.block3.unit24.relu', 'body.block3.unit25.relu',
    'body.block3.unit26.relu', 'body.block3.unit27.relu',
    'body.block3.unit28.relu', 'body.block3.unit29.relu',
    'body.block3.unit30.relu', 'body.block3.unit31.relu',
    'body.block3.unit32.relu', 'body.block3.unit33.relu',
    'body.block3.unit34.relu', 'body.block3.unit35.relu',
    'body.block3.unit36.relu', 'body.block4.unit01.relu',
    'body.block4.unit02.relu', 'body.block4.unit03.relu'
]


def get_model_list():
    return ALL_MODELS

def get_weights(bit_variant):
    response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))

def get_model(name):
    assert name in ALL_MODELS
    if name.startswith('BiT-S-'):
        model = models.KNOWN_MODELS[name](head_size=1000)     # Small BiTs are pretrained on ImageNet
    elif name.startswith('BiT-M-'):
        model = models.KNOWN_MODELS[name]()    # Medium BiTs are trained on ImageNet-21K
    weights = get_weights(name)    
    model.load_from(weights)
    model.eval()
    image_size = 224
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)    
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name in ALL_MODELS
    if 'R50' in name:
        return R50_LAYERS
    elif 'R101' in name:
        return R101_LAYERS
    elif 'R152' in name:
        return R152_LAYERS

def get_bibtex(model_identifier):
    return """@article{touvron2020deit,
    title={Training data-efficient image transformers & distillation through attention},
    author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
    journal={arXiv preprint arXiv:2012.12877},
    year={2020}
    }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
