from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cv_18_dagger_408_pretrained'] = ModelCommitment(identifier='cv_18_dagger_408_pretrained', activations_model=get_model('cv_18_dagger_408_pretrained'), layers=get_layers('cv_18_dagger_408_pretrained'))
