from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eMMCR_Vanilla"] = lambda: ModelCommitment(
    identifier="eMMCR_Vanilla",
    activations_model=get_model("eMMCR_Vanilla"),
    layers=get_layers("eMMCR_Vanilla"),
)
