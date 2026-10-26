"""
Add plugin to the brainscore_vision package.
"""

from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry["vith14_untrained_linhead"] = lambda: ModelCommitment(
    identifier="vith14_untrained_linhead",
    activations_model=get_model("vith14_untrained_linhead"),
    layers=get_layers("vith14_untrained_linhead"),
    behavioral_readout_layer="backbone.fc",
)
