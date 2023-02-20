from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb2_456_406'] = ModelCommitment(identifier='effnetb2_456_406', activations_model=get_model('effnetb2_456_406'), layers=get_layers('effnetb2_456_406'))
