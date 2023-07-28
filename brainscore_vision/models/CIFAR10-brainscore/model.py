import logging
import sys
from typing import List

import numpy as np
from torchvision import transforms as T

from cifar10_models import download_weights_if_needed
from cifar10_models.resnet import resnet50
from model_tools.activations import PytorchWrapper
from model_tools.activations.pytorch import load_images
from model_tools.check_submission import check_models


def get_model_list() -> List[str]:
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    identifiers = [
        "resnet50-cifar",
    ]
    return identifiers


def get_model(identifier: str) -> PytorchWrapper:
    download_weights_if_needed()

    assert identifier == 'resnet50-cifar'
    model = resnet50(pretrained=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    image_size = 32
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
            lambda img: img.unsqueeze(0),
        ]
    )

    def load_preprocess_images(image_filepaths):
        images = load_images(image_filepaths)
        images = [transform(image) for image in images]
        images = np.concatenate(images)
        return images

    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=load_preprocess_images)
    return wrapper


def get_layers(identifier: str) -> List[str]:
    assert identifier == 'resnet50-cifar'
    return ['maxpool'] + \
        [f"layer{layer + 1}.{bottleneck}"
         for layer, num_bottlenecks in enumerate([3, 4, 6, 3]) for bottleneck in range(num_bottlenecks)] + \
        ['avgpool']


def get_bibtex(identifier: str) -> str:
    return """@software{huy_phan_2021_4431043,
      author       = {Huy Phan},
      title        = {huyvnphan/PyTorch\_CIFAR10},
      month        = jan,
      year         = 2021,
      publisher    = {Zenodo},
      version      = {v3.0.1},
      doi          = {10.5281/zenodo.4431043},
      url          = {https://doi.org/10.5281/zenodo.4431043}
    }"""


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['PIL', 'matplotlib', 's3transfer', 'botocore', 'boto3']:
        logging.getLogger(shush_logger).setLevel(logging.INFO)
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
