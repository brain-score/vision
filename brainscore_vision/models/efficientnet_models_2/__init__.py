from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['efficientnet-b0'] = ModelCommitment(identifier='efficientnet-b0', activations_model=get_model('efficientnet-b0'), layers=get_layers('efficientnet-b0'))
model_registry['efficientnet-b2'] = ModelCommitment(identifier='efficientnet-b2', activations_model=get_model('efficientnet-b2'), layers=get_layers('efficientnet-b2'))
model_registry['efficientnet-b4'] = ModelCommitment(identifier='efficientnet-b4', activations_model=get_model('efficientnet-b4'), layers=get_layers('efficientnet-b4'))
model_registry['efficientnet-b6'] = ModelCommitment(identifier='efficientnet-b6', activations_model=get_model('efficientnet-b6'), layers=get_layers('efficientnet-b6'))
