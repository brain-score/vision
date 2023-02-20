from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['sparse-layer-model'] = ModelCommitment(identifier='sparse-layer-model', activations_model=get_model('sparse-layer-model'), layers=get_layers('sparse-layer-model'))
