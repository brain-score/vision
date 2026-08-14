from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

# Explicit behavioural readout: ModelCommitment otherwise defaults to layers[-1],
# so the choice would silently depend on list order.
model_registry['seresnext50_32x4d'] = lambda: ModelCommitment(
    identifier='seresnext50_32x4d',
    activations_model=get_model('seresnext50_32x4d'),
    layers=get_layers('seresnext50_32x4d'),
    behavioral_readout_layer='fc',
)
