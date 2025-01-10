from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['BT_CORnet-S-1001'] = lambda: ModelCommitment(identifier='BT_CORnet-S-1001', activations_model=get_model('BT_CORnet-S-1001'), layers=get_layers('BT_CORnet-S-1001'))