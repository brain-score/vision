from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['crossvit_18_dagger_408_FAT_rotinv'] = lambda: ModelCommitment(
    identifier='crossvit_18_dagger_408_FAT_rotinv',
    activations_model=get_model(),
    layers=LAYERS
)
