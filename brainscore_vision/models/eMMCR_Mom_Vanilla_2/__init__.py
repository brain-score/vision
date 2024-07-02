from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eMMCR_Mom_Vanilla_2"] = lambda: ModelCommitment(
    identifier="eMMCR_Mom_Vanilla_2",
    activations_model=get_model("eMMCR_Mom_Vanilla_2"),
    layers=get_layers("eMMCR_Mom_Vanilla_2"),
)
