"""
Add plugin to the brainscore_vision package.
"""

from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry["ijepa_vits14_linhead_400ep"] = lambda: ModelCommitment(
    identifier="ijepa_vits14_linhead_400ep",
    activations_model=get_model("ijepa_vits14_linhead_400ep"),
    layers=get_layers("ijepa_vits14_linhead_400ep"),
    behavioral_readout_layer="backbone.fc",
)
