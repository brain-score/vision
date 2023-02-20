from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['SWSL_resnet50'] = ModelCommitment(identifier='SWSL_resnet50', activations_model=get_model('SWSL_resnet50'), layers=get_layers('SWSL_resnet50'))
