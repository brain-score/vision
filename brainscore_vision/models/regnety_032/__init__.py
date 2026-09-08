from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

# Explicit behavioural readout: ModelCommitment otherwise defaults to layers[-1],
# so the choice would silently depend on list order.
model_registry['regnety_032'] = lambda: ModelCommitment(
    identifier='regnety_032',
    activations_model=get_model('regnety_032'),
    layers=get_layers('regnety_032'),
    behavioral_readout_layer='head',
)
