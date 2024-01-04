from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetv2retrain'] = ModelCommitment(identifier='effnetv2retrain', activations_model=get_model('effnetv2retrain'), layers=get_layers('effnetv2retrain'))
