from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['crossvit_18_dagger_408_FAT_rotinv'] = ModelCommitment(identifier='crossvit_18_dagger_408_FAT_rotinv', activations_model=get_model('crossvit_18_dagger_408_FAT_rotinv'), layers=get_layers('crossvit_18_dagger_408_FAT_rotinv'))
