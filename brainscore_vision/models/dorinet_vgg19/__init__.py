from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['Dorinet_vgg'] = ModelCommitment(identifier='Dorinet_vgg', activations_model=get_model('Dorinet_vgg'), layers=get_layers('Dorinet_vgg'))
