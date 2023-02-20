import logging
import sys

import torch
import torchvision.transforms as T
from PIL import Image

from model_tools.activations import PytorchWrapper
from model_tools.check_submission import check_models

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
    return [
        "omnivore_swinT",
        "omnivore_swinS",
        "omnivore_swinB",
        "omnivore_swinB_imagenet21k",
        "omnivore_swinL_imagenet21k",
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
    model = torch.hub.load("facebookresearch/omnivore", model=name)
    model.eval()
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing,
                             forward_kwargs={'input_type': 'image'})
    wrapper.image_size = 224
    return wrapper


def preprocessing(stimuli_paths):
    # following https://github.com/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb
    stimuli = [Image.open(image_path).convert("RGB") for image_path in stimuli_paths]
    image_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    stimuli = [image_transform(image) for image in stimuli]
    # The model expects inputs of shape: B x C x T x H x W
    stimuli = [image[:, None, ...] for image in stimuli]
    return stimuli


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
    model_blocks = {
        "omnivore_swinT": [2, 2, 6, 2],
        "omnivore_swinS": [2, 2, 18, 2],
        "omnivore_swinB": [2, 2, 18, 2],
        "omnivore_swinB_imagenet21k": [2, 2, 18, 2],
        "omnivore_swinL_imagenet21k": [2, 2, 18, 2],
    }
    blocks = model_blocks[name]
    return ['trunk.pos_drop'] + [f'trunk.layers.{layer}.blocks.{block}'
                                 for layer, num_blocks in enumerate(blocks) for block in range(num_blocks)]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{girdhar2022omnivore,
              title={{Omnivore: A Single Model for Many Visual Modalities}},
              author={Girdhar, Rohit and Singh, Mannat and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
              booktitle={CVPR},
              year={2022},
              url={https://doi.org/10.48550/arXiv.2201.08377}
            }"""


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['PIL', 'brainscore.metrics']:
        logging.getLogger(shush_logger).setLevel(logging.INFO)
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
