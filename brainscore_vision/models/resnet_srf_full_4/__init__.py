from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_srf_full_apr19_tinyImagenet'] = ModelCommitment(identifier='resnet_srf_full_apr19_tinyImagenet', activations_model=get_model('resnet_srf_full_apr19_tinyImagenet'), layers=get_layers('resnet_srf_full_apr19_tinyImagenet'))
