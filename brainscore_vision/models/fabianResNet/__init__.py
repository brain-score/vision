from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['fabianResNet'] = lambda: ModelCommitment(identifier='fabianResNet', activations_model=get_model('fabianResNet'), layers=get_layers('fabianResNet'))