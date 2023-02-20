from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CLIP-RN50'] = ModelCommitment(identifier='CLIP-RN50', activations_model=get_model('CLIP-RN50'), layers=get_layers('CLIP-RN50'))
