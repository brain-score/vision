from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['model-lecs-v1.0.1'] = lambda: ModelCommitment(identifier='model-lecs-v1.0.1', activations_model=get_model('model-lecs-v1.0.1'), layers=get_layers('model-lecs-v1.0.1'))