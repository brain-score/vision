from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['cv_18_dagger_408_pretrained'] = lambda: ModelCommitment(identifier='cv_18_dagger_408_pretrained', activations_model=get_model(), layers=LAYERS)
