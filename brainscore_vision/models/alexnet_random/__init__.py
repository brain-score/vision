from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_random'] = lambda: ModelCommitment(identifier='alexnet_random',
                                                              activations_model=get_model('alexnet_random'), 
                                                              layers=get_layers('alexnet_random'))
