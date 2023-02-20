from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_z_aacn_2'] = ModelCommitment(identifier='cornet_z_aacn_2', activations_model=get_model('cornet_z_aacn_2'), layers=get_layers('cornet_z_aacn_2'))
