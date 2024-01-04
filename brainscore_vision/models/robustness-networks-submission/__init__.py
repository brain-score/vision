from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['konkle_alexnetgn_ipcl_ref01_primary_model'] = ModelCommitment(identifier='konkle_alexnetgn_ipcl_ref01_primary_model', activations_model=get_model('konkle_alexnetgn_ipcl_ref01_primary_model'), layers=get_layers('konkle_alexnetgn_ipcl_ref01_primary_model'))
