from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .helpers.cornet_helpers import vonecornet


def get_model(name):
    assert name == 'VOneCORnet-S'
    return vonecornet('cornets')

def get_layers(name):
    assert name == 'VOneCORnet-S'
    return ['vone_block.output-t0'] + [f'model.{area}.output-t{timestep}'
                                                                   for area, timesteps in
                                                                   [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                                                   for timestep in timesteps] + ['model.decoder.avgpool-t0']

def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
