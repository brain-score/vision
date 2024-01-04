import functools

import numpy as np
from model_tools.activations.pytorch import PytorchWrapper, load_images
from model_tools.check_submission import check_models
from torchvision import transforms as T

from r3m import load_r3m

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ["r3m_resnet50_nocrop",
            "r3m_resnet34_nocrop",
            "r3m_resnet18_nocrop"
    ]


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    mapping = {"r3m_resnet50_nocrop": "resnet50",
               "r3m_resnet34_nocrop": "resnet34",
               "r3m_resnet18_nocrop": "resnet18"}
    modelid = mapping[name]
    model = load_r3m(modelid).module

    image_size = 224
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def load_preprocess_images(image_filepaths, image_size):
    """
    define custom pre-processing here since R3M does not normalize like other models
    :seealso: r3m/example.py
    """
    images = load_images(image_filepaths)
    # preprocessing
    transforms = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # ToTensor() divides by 255
        lambda img: img.unsqueeze(0),
    ])
    images = [transforms(image) * 255.0 for image in images]  # R3M expects image input to be [0-255]
    images = np.concatenate(images)
    return images


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    units_mapping = {"r3m_resnet50_nocrop": [3, 4, 6, 3],
                     "r3m_resnet34_nocrop": [3, 4, 6, 3],
                     "r3m_resnet18_nocrop": [2, 2, 2, 2]}
    units = units_mapping[name]
    prefix = "convnet"
    layers = [f'{prefix}.conv1'] + \
             [f'{prefix}.layer{layer_num}.{bottleneck}.relu'
              for layer_num, bottlenecks in enumerate(units, start=1)
              for bottleneck in range(bottlenecks)] + \
             [f'{prefix}.avgpool', f'{prefix}.fc']
    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """
    @article{https://doi.org/10.48550/arxiv.2203.12601,
      doi = {10.48550/ARXIV.2203.12601},
      url = {https://arxiv.org/abs/2203.12601},
      author = {Nair, Suraj and Rajeswaran, Aravind and Kumar, Vikash and Finn, Chelsea and Gupta, Abhinav},
      keywords = {Robotics (cs.RO), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {R3M: A Universal Visual Representation for Robot Manipulation},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
