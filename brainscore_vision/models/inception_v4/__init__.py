from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['inception_v4'] = lambda: ModelCommitment(identifier='inception_v4',
                                                               activations_model=get_model('inception_v4'),
                                                               layers=get_layers('inception_v4'))
