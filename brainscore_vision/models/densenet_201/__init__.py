from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['densenet-201'] = lambda: ModelCommitment(identifier='densenet-201',
                                                               activations_model=get_model('densenet-201'),
                                                               layers=get_layers('densenet-201'))
