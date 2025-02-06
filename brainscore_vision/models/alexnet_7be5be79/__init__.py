from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_7be5be79'] = lambda: ModelCommitment(identifier='alexnet_7be5be79',
                                                              activations_model=get_model('alexnet_7be5be79'), 
                                                              layers=get_layers('alexnet_7be5be79'))
