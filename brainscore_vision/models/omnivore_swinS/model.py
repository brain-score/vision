import functools
from brainscore_vision.model_helpers.activations.pytorch import load_images, preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch
import numpy as np

def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    images = images[:, :, None, ...]
    return images

model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinS", force_reload=True)

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'omnivore_swinS'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='name', model=model, preprocessing=preprocessing,forward_kwargs={"input_type":"image"})
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'omnivore_swinS'
    layer_names = []
    for name, module in model.named_modules():
        layer_names.append(name)

    return layer_names[-50:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/abs-2201-08377,
  author       = {Rohit Girdhar and
                  Mannat Singh and
                  Nikhila Ravi and
                  Laurens van der Maaten and
                  Armand Joulin and
                  Ishan Misra},
  title        = {Omnivore: {A} Single Model for Many Visual Modalities},
  journal      = {CoRR},
  volume       = {abs/2201.08377},
  year         = {2022},
  url          = {https://arxiv.org/abs/2201.08377},
  eprinttype    = {arXiv},
  eprint       = {2201.08377},
  timestamp    = {Tue, 01 Feb 2022 14:59:01 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2201-08377.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)