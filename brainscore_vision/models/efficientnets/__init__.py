from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['AdvProp_efficientnet-b0'] = ModelCommitment(identifier='AdvProp_efficientnet-b0', activations_model=get_model('AdvProp_efficientnet-b0'), layers=get_layers('AdvProp_efficientnet-b0'))
model_registry['AdvProp_efficientnet-b2'] = ModelCommitment(identifier='AdvProp_efficientnet-b2', activations_model=get_model('AdvProp_efficientnet-b2'), layers=get_layers('AdvProp_efficientnet-b2'))
model_registry['AdvProp_efficientnet-b4'] = ModelCommitment(identifier='AdvProp_efficientnet-b4', activations_model=get_model('AdvProp_efficientnet-b4'), layers=get_layers('AdvProp_efficientnet-b4'))
model_registry['AdvProp_efficientnet-b6'] = ModelCommitment(identifier='AdvProp_efficientnet-b6', activations_model=get_model('AdvProp_efficientnet-b6'), layers=get_layers('AdvProp_efficientnet-b6'))
model_registry['AdvProp_efficientnet-b7'] = ModelCommitment(identifier='AdvProp_efficientnet-b7', activations_model=get_model('AdvProp_efficientnet-b7'), layers=get_layers('AdvProp_efficientnet-b7'))
model_registry['AdvProp_efficientnet-b8'] = ModelCommitment(identifier='AdvProp_efficientnet-b8', activations_model=get_model('AdvProp_efficientnet-b8'), layers=get_layers('AdvProp_efficientnet-b8'))
