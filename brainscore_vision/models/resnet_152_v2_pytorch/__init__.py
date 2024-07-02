from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-152_v2_pytorch'] = lambda: ModelCommitment(identifier='resnet-152_v2_pytorch',
                                                               activations_model=get_model('resnet-152_v2_pytorch'),
                                                               layers=get_layers('resnet-152_v2_pytorch'))
