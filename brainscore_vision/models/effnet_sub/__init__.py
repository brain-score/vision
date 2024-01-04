from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb0_retrain'] = ModelCommitment(identifier='effnetb0_retrain', activations_model=get_model('effnetb0_retrain'), layers=get_layers('effnetb0_retrain'))
