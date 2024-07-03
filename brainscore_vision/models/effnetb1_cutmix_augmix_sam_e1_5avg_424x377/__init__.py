from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_cutmix_augmix_sam_e1_5avg_424x377'] = lambda: ModelCommitment(
    identifier='effnetb1_cutmix_augmix_sam_e1_5avg_424x377',
    activations_model=get_model('effnetb1_cutmix_augmix_sam_e1_5avg_424x377'),
    layers=get_layers('effnetb1_cutmix_augmix_sam_e1_5avg_424x377')
)