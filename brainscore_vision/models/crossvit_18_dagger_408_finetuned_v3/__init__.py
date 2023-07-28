from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['custom_model_cv_18_dagger_408_v3'] = ModelCommitment(identifier='custom_model_cv_18_dagger_408_v3', activations_model=get_model('custom_model_cv_18_dagger_408_v3'), layers=get_layers('custom_model_cv_18_dagger_408_v3'))
