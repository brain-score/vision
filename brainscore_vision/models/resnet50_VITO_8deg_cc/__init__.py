from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers, get_model


model_registry['resnet50-VITO-8deg-cc'] = lambda: ModelCommitment(identifier='resnet50-VITO-8deg-cc',
                                                                    activations_model=get_model('resnet50-VITO-8deg-cc'),
                                                                    layers=get_layers('resnet50-VITO-8deg-cc'))

