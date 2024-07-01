from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['pnasnet_large_pytorch'] = lambda: ModelCommitment(identifier='pnasnet_large_pytorch',
                                                               activations_model=get_model('pnasnet_large_pytorch'),
                                                               layers=get_layers('pnasnet_large_pytorch'))