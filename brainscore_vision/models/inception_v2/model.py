import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch.utils.model_zoo as model_zoo
import timm
import torch.nn as nn

def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'inception_v2'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model  = timm.create_model('inception_resnet_v2', pretrained=True)
    wrapper = PytorchWrapper(identifier='inception_v2', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'inception_v2'
    model  = timm.create_model('inception_resnet_v2', pretrained=True)
    return list(dict(model.named_modules()).keys())[1:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@misc{szegedy2016inceptionv4,
      title={Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning}, 
      author={Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke and Alex Alemi},
      year={2016},
      eprint={1602.07261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
