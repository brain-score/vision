from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["r50-e10-cut1"] = lambda: ModelCommitment(
    identifier="r50-e10-cut1",
    activations_model=get_model("r50-e10-cut1"),
    layers=LAYERS)

model_registry["r50-e20-cut1"] = lambda: ModelCommitment(
    identifier="r50-e10-cut1",
    activations_model=get_model("r50-e20-cut1"),
    layers=LAYERS)

model_registry["r50-e35-cut1"] = lambda: ModelCommitment(
    identifier="r50-e10-cut1",
    activations_model=get_model("r50-e35-cut1"),
    layers=LAYERS)

model_registry["r50-e50-cut1"] = lambda: ModelCommitment(
    identifier="r50-e50-cut1",
    activations_model=get_model("r50-e50-cut1"),
    layers=LAYERS)

model_registry["imgnfull-e45-cut1"] = lambda: ModelCommitment(
    identifier="imgnfull-e45-cut1",
    activations_model=get_model("imgnfull-e45-cut1"),
    layers=LAYERS)

model_registry["imgnfull-e60-cut1"] = lambda: ModelCommitment(
    identifier="imgnfull-e60-cut1",
    activations_model=get_model("imgnfull-e60-cut1"),
    layers=LAYERS)
