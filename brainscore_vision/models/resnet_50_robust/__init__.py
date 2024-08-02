from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-50-robust'] = lambda: ModelCommitment(identifier='resnet-50-robust',
                                                               activations_model=get_model('resnet-50-robust'),
                                                               layers=get_layers('resnet-50-robust'))
