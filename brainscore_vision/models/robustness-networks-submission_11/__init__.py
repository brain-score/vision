from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['CLIP_resnet50'] = ModelCommitment(identifier='CLIP_resnet50', activations_model=get_model('CLIP_resnet50'), layers=get_layers('CLIP_resnet50'))
