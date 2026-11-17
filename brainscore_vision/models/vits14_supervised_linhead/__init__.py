"""
Add plugin to the brainscore_vision package.
"""

from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry["vits14_supervised_linhead"] = lambda: ModelCommitment(
    identifier="vits14_supervised_linhead",
    activations_model=get_model("vits14_supervised_linhead"),
    layers=get_layers("vits14_supervised_linhead"),
    behavioral_readout_layer="backbone.fc",
)
