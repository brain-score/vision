from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['vone_resnet1'] = ModelCommitment(identifier='vone_resnet1', activations_model=get_model('vone_resnet1'), layers=get_layers('vone_resnet1'))
