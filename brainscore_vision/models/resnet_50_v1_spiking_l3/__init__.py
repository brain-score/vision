from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_50_v1_spiking_l3'] = lambda: ModelCommitment(identifier='resnet_50_v1_spiking_l3', activations_model=get_model('resnet_50_v1_spiking_l3'), layers=get_layers('resnet_50_v1_spiking_l3'))