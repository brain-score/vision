import sys
import torch
import timm
assert timm.__version__ == "0.3.2"
import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models

sys.path.append('../')
import deit_models as models


ALL_MODELS_FUNC_DICT = {
    'deit_tiny_patch16_224_id': 'deit_tiny_patch16_224',
    'deit_small_patch16_224_id': 'deit_small_patch16_224',
    'deit_base_patch16_224_id': 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224_id': 'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224_id': 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224_id': 'deit_base_distilled_patch16_224',
    'deit_base_patch16_384_id': 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384_id': 'deit_base_distilled_patch16_384',
}
ALL_MODELS = list(ALL_MODELS_FUNC_DICT.keys())


def get_model_list():
    return ALL_MODELS

def get_model(name):
    assert name in ALL_MODELS
    func_name = ALL_MODELS_FUNC_DICT[name]
    model_func = getattr(models, func_name)
    model = model_func(pretrained=True)
    model.eval()
    image_size = int(func_name.split('_')[-1])
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)    
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name in ALL_MODELS
    num_of_blocks = 12
    return [f'blocks.{i}.mlp.fc2' for i in range(num_of_blocks)]

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
    
