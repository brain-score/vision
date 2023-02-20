from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_vonemap_retrain'] = ModelCommitment(identifier='effnetb1_vonemap_retrain', activations_model=get_model('effnetb1_vonemap_retrain'), layers=get_layers('effnetb1_vonemap_retrain'))
