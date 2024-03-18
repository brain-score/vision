from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS


model_registry['googlenet'] = lambda: ModelCommitment(
    identifier='googlenet', activations_model=get_model(), layers=LAYERS)
