import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torchvision


model = torchvision.models.resnet50(weights='DEFAULT')

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet-50-pytorch'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet-50-pytorch'
    return list(dict(model.named_modules()).keys())[1:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/HeZRS15,
  author       = {Kaiming He and
                  Xiangyu Zhang and
                  Shaoqing Ren and
                  Jian Sun},
  title        = {Deep Residual Learning for Image Recognition},
  journal      = {CoRR},
  volume       = {abs/1512.03385},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.03385},
  eprinttype    = {arXiv},
  eprint       = {1512.03385},
  timestamp    = {Wed, 25 Jan 2023 11:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
