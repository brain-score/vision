from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['BT_CORnet-S-100'] =lambda: ModelCommitment(identifier='BT_CORnet-S-100', activations_model=get_model('BT_CORnet-S-100'), layers=get_layers('BT_CORnet-S-100'))