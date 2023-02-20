from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['sketch_model_12-ep25'] = ModelCommitment(identifier='sketch_model_12-ep25', activations_model=get_model('sketch_model_12-ep25'), layers=get_layers('sketch_model_12-ep25'))
