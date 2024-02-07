from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['tv_efficientnet-b1'] = lambda: ModelCommitment(identifier='tv_efficientnet-b1', activations_model=get_model('tv_efficientnet-b1'), layers=get_layers('tv_efficientnet-b1'))
