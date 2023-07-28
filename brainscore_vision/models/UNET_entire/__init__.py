from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['unet_entire'] = ModelCommitment(identifier='unet_entire', activations_model=get_model('unet_entire'), layers=get_layers('unet_entire'))
