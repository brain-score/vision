from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_109'] = lambda: ModelCommitment(identifier='grcnn_109', activations_model=get_model('grcnn_109'), layers=get_layers('grcnn_109'))
