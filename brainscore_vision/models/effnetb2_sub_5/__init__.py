from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb2_cutmix_augmix_epoch3_348x309'] = ModelCommitment(identifier='effnetb2_cutmix_augmix_epoch3_348x309', activations_model=get_model('effnetb2_cutmix_augmix_epoch3_348x309'), layers=get_layers('effnetb2_cutmix_augmix_epoch3_348x309'))
