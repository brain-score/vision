import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models


device = "cpu"


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
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='shufflenet_v2_x1_0', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'shufflenet_v2_x1_0'
    return ['conv1', 'stage2', 'stage3', 'stage4', 'conv5', 'fc']


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
            """@article{DBLP:journals/corr/abs-1807-11164,
                            title = {ShuffleNet {V2:} Practical Guidelines for Efficient {CNN} Architecture Design},
                            author = {Ningning Ma, Xiangyu Zhang, Hai{-}Tao Zheng and Jian Sun},
                            journal   = {CoRR},
                            volume    = {abs/1807.11164},
                            year = {2018},
                            url = {http://arxiv.org/abs/1807.11164}
                            }"""
            '''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
