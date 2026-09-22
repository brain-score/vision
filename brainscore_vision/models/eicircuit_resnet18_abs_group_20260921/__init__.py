from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_model, get_layers


model_registry['eicircuit_resnet18_abs_group_20260921'] = lambda: ModelCommitment(
    identifier='eicircuit_resnet18_abs_group_20260921',
    activations_model=get_model('eicircuit_resnet18_abs_group_20260921'),
    layers=get_layers('eicircuit_resnet18_abs_group_20260921'),
    behavioral_readout_layer='features.avgpool',
    visual_degrees=8,
)
