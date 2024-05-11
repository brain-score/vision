from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_spcl'] = lambda: ModelCommitment(identifier='resnet18_spcl', activations_model=get_model('resnet18_spcl'), layers=get_layers('resnet18_spcl'))
