from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_batch512_3steps_eps0_layer_x'] = ModelCommitment(identifier='resnet50_batch512_3steps_eps0_layer_x', activations_model=get_model('resnet50_batch512_3steps_eps0_layer_x'), layers=get_layers('resnet50_batch512_3steps_eps0_layer_x'))
