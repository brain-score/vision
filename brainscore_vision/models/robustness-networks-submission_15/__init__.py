from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['SWSL_resnext101_32x8d'] = ModelCommitment(identifier='SWSL_resnext101_32x8d', activations_model=get_model('SWSL_resnext101_32x8d'), layers=get_layers('SWSL_resnext101_32x8d'))
