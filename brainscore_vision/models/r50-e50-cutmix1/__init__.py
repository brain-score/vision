from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['r50-e50-cut1'] = ModelCommitment(identifier='r50-e50-cut1', activations_model=get_model('r50-e50-cut1'), layers=get_layers('r50-e50-cut1'))
