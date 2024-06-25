from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['model-lecs'] = lambda: ModelCommitment(identifier='model-lecs', activations_model=get_model('model-lecs'), layers=get_layers('model-lecs'))