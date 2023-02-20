from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb2_cutmix_augmix_epoch3_424x377'] = ModelCommitment(identifier='effnetb2_cutmix_augmix_epoch3_424x377', activations_model=get_model('effnetb2_cutmix_augmix_epoch3_424x377'), layers=get_layers('effnetb2_cutmix_augmix_epoch3_424x377'))
