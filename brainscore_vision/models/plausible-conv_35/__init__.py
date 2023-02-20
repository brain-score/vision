from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['efficientnet_b1_untrained'] = ModelCommitment(identifier='efficientnet_b1_untrained', activations_model=get_model('efficientnet_b1_untrained'), layers=get_layers('efficientnet_b1_untrained'))
