"""
Add plugin to the brainscore_vision package.
"""

from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model

model_registry["resnet50_byol_linhead"] = lambda: ModelCommitment(
    identifier="resnet50_byol_linhead",
    activations_model=get_model("resnet50_byol_linhead"),
    layers=get_layers("resnet50_byol_linhead"),
    behavioral_readout_layer="backbone.fc",
)
