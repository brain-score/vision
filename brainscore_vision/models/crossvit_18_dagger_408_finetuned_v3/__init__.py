from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['custom_model_cv_18_dagger_408_v3'] = lambda: ModelCommitment(
    identifier='custom_model_cv_18_dagger_408_v3',
    activations_model=get_model(),
    layers=LAYERS,
)
