from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['FrankRobWobv0'] = ModelCommitment(identifier='FrankRobWobv0', activations_model=get_model('FrankRobWobv0'), layers=get_layers('FrankRobWobv0'))
