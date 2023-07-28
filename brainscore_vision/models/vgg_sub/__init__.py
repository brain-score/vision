from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['repvgg_b3'] = ModelCommitment(identifier='repvgg_b3', activations_model=get_model('repvgg_b3'), layers=get_layers('repvgg_b3'))
