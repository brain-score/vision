import functools
from torchvision.models import densenet121
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'densenet-121'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = densenet121(weights='DEFAULT')
    wrapper = PytorchWrapper(identifier='densenet-121', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'densenet-121'
    layer_names = (['norm0'] + ['pool0'] +
                   [f'denseblock1.denselayer{i}' for i in range(1, 7)] + ['transition1.pool'] +
                   [f'denseblock2.denselayer{i}' for i in range(1, 13)] + ['transition2.pool'] +
                   [f'denseblock3.denselayer{i}' for i in range(1, 25)] + ['transition3.pool'] +
                   [f'denseblock4.denselayer{i}' for i in range(1, 17)])
    layer_names = [f"features.{name}" for name in layer_names]
    layer_names += ['features.norm5']
    return layer_names


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/HuangLW16a,
  author       = {Gao Huang and
                  Zhuang Liu and
                  Kilian Q. Weinberger},
  title        = {Densely Connected Convolutional Networks},
  journal      = {CoRR},
  volume       = {abs/1608.06993},
  year         = {2016},
  url          = {http://arxiv.org/abs/1608.06993},
  eprinttype    = {arXiv},
  eprint       = {1608.06993},
  timestamp    = {Mon, 10 Sep 2018 15:49:32 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/HuangLW16a.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)