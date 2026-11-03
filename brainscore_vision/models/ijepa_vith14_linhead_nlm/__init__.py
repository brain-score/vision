"""
Add plugin to the brainscore_vision package.
"""

from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry["ijepa_vith14_linhead_nlm"] = lambda: ModelCommitment(
    identifier="ijepa_vith14_linhead_nlm",
    activations_model=get_model("ijepa_vith14_linhead_nlm"),
    layers=get_layers("ijepa_vith14_linhead_nlm"),
    behavioral_readout_layer="backbone.fc",
)
