from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['CoupNet2'] = lambda: ModelCommitment(identifier='CoupNet2',
                                                              activations_model=get_model('CoupNet2'),
                                                              layers=get_layers('CoupNet2'))
