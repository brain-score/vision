from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_batch512_3steps_eps0'] = ModelCommitment(identifier='resnet50_batch512_3steps_eps0', activations_model=get_model('resnet50_batch512_3steps_eps0'), layers=get_layers('resnet50_batch512_3steps_eps0'))
