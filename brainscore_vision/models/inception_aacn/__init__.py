from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['inception_aacn'] = ModelCommitment(identifier='inception_aacn', activations_model=get_model('inception_aacn'), layers=get_layers('inception_aacn'))
