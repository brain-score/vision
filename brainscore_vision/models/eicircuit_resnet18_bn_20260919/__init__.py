from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_model, get_layers


model_registry['eicircuit_resnet18_bn_20260919'] = lambda: ModelCommitment(
    identifier='eicircuit_resnet18_bn_20260919',
    activations_model=get_model('eicircuit_resnet18_bn_20260919'),
    layers=get_layers('eicircuit_resnet18_bn_20260919'),
    behavioral_readout_layer='features.avgpool',
    visual_degrees=8,
)
