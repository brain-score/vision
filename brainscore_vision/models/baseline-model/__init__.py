from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cb-baseline-model'] = ModelCommitment(identifier='cb-baseline-model', activations_model=get_model('cb-baseline-model'), layers=get_layers('cb-baseline-model'))
