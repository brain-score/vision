from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['cornet_z_reg'] = lambda: ModelCommitment(identifier='cornet_z_reg', activations_model=get_model('cornet_z_reg'), layers=get_layers('cornet_z_reg'))
