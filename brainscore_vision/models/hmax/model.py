import functools
import os

from model_tools.check_submission import check_models

from models.hmax import HMAX
from models.pytorch import PytorchWrapper

"""
Module for hmax submission to brain-score
"""


def get_model_list():
    # return ['hmax' , 'hmax_s2']
    return ['hmax']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    return get_hmax(name, 224)


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
    return ['s1_out', 'c1_out', 'c2_out', 's2_out']
    # return ['s1_out', 'c1_out', 'c2_out'] if name == 'hmax' else ['s2_out']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@ARTICLE {,
                author = {G. Cortelazzo and M. Balanza},
                journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
                title = {Frequency Domain Analysis of Translations with Piecewise Cubic Trajectories},
                year = {1993},
                volume = {29},
                number = {04},
                issn = {1939-3539},
                pages = {411-416},
                keywords = {frequency domain motion analysis; motion estimation; translations; piecewise cubic trajectories; cubic spline trajectories; finite-duration effects; constant velocity motion; first-order model; frequency-domain analysis; motion estimation; splines (mathematics)},
                doi = {10.1109/34.206960},
                publisher = {IEEE Computer Society},
                address = {Los Alamitos, CA, USA},
                month = {apr}
                }
            """


def get_hmax(identifier, image_size):
    path = os.path.join(os.path.dirname(__file__), 'universal_patch_set.mat')
    model = HMAX(path)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model,
                             preprocessing=preprocessing, batch_size=2)
    wrapper.image_size = image_size
    return wrapper


if __name__ == '__main__':
    # Use this method to ensure the correctness of the  BaeeModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
