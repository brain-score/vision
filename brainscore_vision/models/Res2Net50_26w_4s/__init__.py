from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['Res2Net50_26w_4s'] = lambda: ModelCommitment(identifier='Res2Net50_26w_4s', activations_model=get_model('Res2Net50_26w_4s'), layers=get_layers('Res2Net50_26w_4s'))
