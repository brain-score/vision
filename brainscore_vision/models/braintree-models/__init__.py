from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnet-S'] = ModelCommitment(identifier='CORnet-S', activations_model=get_model('CORnet-S'), layers=get_layers('CORnet-S'))
model_registry['BT_CORnet-S-100'] = ModelCommitment(identifier='BT_CORnet-S-100', activations_model=get_model('BT_CORnet-S-100'), layers=get_layers('BT_CORnet-S-100'))
model_registry['BT_CORnet-S-500'] = ModelCommitment(identifier='BT_CORnet-S-500', activations_model=get_model('BT_CORnet-S-500'), layers=get_layers('BT_CORnet-S-500'))
model_registry['BT_CORnet-S-1001'] = ModelCommitment(identifier='BT_CORnet-S-1001', activations_model=get_model('BT_CORnet-S-1001'), layers=get_layers('BT_CORnet-S-1001'))
