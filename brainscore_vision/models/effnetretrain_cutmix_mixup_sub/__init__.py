from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['effnetb0_cutmix_mixup_epoch5'] = lambda: ModelCommitment(
    identifier='effnetb0_cutmix_mixup_epoch5',
    activations_model=get_model(),
    layers=LAYERS)
