from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb0_cutmix_mixup_epoch5'] = ModelCommitment(identifier='effnetb0_cutmix_mixup_epoch5', activations_model=get_model('effnetb0_cutmix_mixup_epoch5'), layers=get_layers('effnetb0_cutmix_mixup_epoch5'))
