from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['custom_model_cv_18_dagger_408_inv_rot'] = lambda: ModelCommitment(
    identifier='custom_model_cv_18_dagger_408_inv_rot',
    activations_model=get_model(),
    layers=LAYERS,
)
