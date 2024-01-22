import torch
import timm
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

from brainscore_vision.model_helpers.check_submission import check_models


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


BIBTEX = """@article{touvron2020deit,
    title={Training data-efficient image transformers & distillation through attention},
    author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
    journal={arXiv preprint arXiv:2012.12877},
    year={2020}
}"""

LAYERS = [f'blocks.{i}' for i in range(12)]


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = torch.hub.load(
            'facebookresearch/deit:main',
            'deit_base_patch16_224',
            pretrained=True).to(device)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier='deit',
        model=model,
        preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
