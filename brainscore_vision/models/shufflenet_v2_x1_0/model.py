import functools
from torchvision.models import shufflenet_v2_x1_0
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'shufflenet_v2_x1_0'
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'shufflenet_v2_x1_0'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = shufflenet_v2_x1_0(weights = 'DEFAULT')
    wrapper = PytorchWrapper(identifier='shufflenet_v2_x1_0', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'shufflenet_v2_x1_0'
    model = shufflenet_v2_x1_0(weights = 'DEFAULT')
    return list(dict(model.named_modules()).keys())[1:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/SzegedyLJSRAEVR14,
  author       = {Christian Szegedy and
                  Wei Liu and
                  Yangqing Jia and
                  Pierre Sermanet and
                  Scott E. Reed and
                  Dragomir Anguelov and
                  Dumitru Erhan and
                  Vincent Vanhoucke and
                  Andrew Rabinovich},
  title        = {Going Deeper with Convolutions},
  journal      = {CoRR},
  volume       = {abs/1409.4842},
  year         = {2014},
  url          = {http://arxiv.org/abs/1409.4842},
  eprinttype    = {arXiv},
  eprint       = {1409.4842},
  timestamp    = {Mon, 13 Aug 2018 16:48:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SzegedyLJSRAEVR14.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
