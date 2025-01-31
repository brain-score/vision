from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['nasnet_large'] = lambda: ModelCommitment(identifier='nasnet_large',
                                                               activations_model=get_model('nasnet_large'),
                                                               layers=get_layers('nasnet_large'))
