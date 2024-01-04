from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_z_aacn_8heads'] = ModelCommitment(identifier='cornet_z_aacn_8heads', activations_model=get_model('cornet_z_aacn_8heads'), layers=get_layers('cornet_z_aacn_8heads'))
