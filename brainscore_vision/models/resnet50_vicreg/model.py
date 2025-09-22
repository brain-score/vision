import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch



def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet50-vicreg'
    model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='name', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet50-vicreg'
    layers = ['conv1', 'layer1.0', 'layer1.1', 'layer1.2',
        'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
        'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3',
         'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2','avgpool']
    return layers


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/abs-2105-04906,
  author       = {Adrien Bardes and
                  Jean Ponce and
                  Yann LeCun},
  title        = {VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised
                  Learning},
  journal      = {CoRR},
  volume       = {abs/2105.04906},
  year         = {2021},
  url          = {https://arxiv.org/abs/2105.04906},
  eprinttype    = {arXiv},
  eprint       = {2105.04906},
  timestamp    = {Fri, 14 May 2021 12:13:30 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2105-04906.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)