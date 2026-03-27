from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['aughost_fair_resnet18'] = lambda: ModelCommitment(identifier='aughost_fair_resnet18', activations_model=get_model('aughost_fair_resnet18'), layers=get_layers('aughost_fair_resnet18'))
