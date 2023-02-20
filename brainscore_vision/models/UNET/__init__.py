from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['unet'] = ModelCommitment(identifier='unet', activations_model=get_model('unet'), layers=get_layers('unet'))
