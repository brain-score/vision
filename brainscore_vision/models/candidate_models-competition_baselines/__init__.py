from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['pixels-baseline'] = ModelCommitment(identifier='pixels-baseline', activations_model=get_model('pixels-baseline'), layers=get_layers('pixels-baseline'))
model_registry['alexnet-baseline'] = ModelCommitment(identifier='alexnet-baseline', activations_model=get_model('alexnet-baseline'), layers=get_layers('alexnet-baseline'))
