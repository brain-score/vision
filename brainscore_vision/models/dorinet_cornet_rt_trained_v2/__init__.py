from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['Dorinet_CORnet_RT_V2'] = ModelCommitment(identifier='Dorinet_CORnet_RT_V2', activations_model=get_model('Dorinet_CORnet_RT_V2'), layers=get_layers('Dorinet_CORnet_RT_V2'))
