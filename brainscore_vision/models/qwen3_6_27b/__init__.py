from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import IDENTIFIER, get_layers, get_model


model_registry['qwen3.6-27b'] = lambda: ModelCommitment(
    identifier=IDENTIFIER,
    activations_model=get_model(IDENTIFIER),
    layers=get_layers(IDENTIFIER),
    behavioral_readout_layer=get_layers(IDENTIFIER)[-1],
)
