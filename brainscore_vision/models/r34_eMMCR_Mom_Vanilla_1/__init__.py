from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["r34_eMMCR_Mom_Vanilla_1"] = lambda: ModelCommitment(
    identifier="r34_eMMCR_Mom_Vanilla_1",
    activations_model=get_model("r34_eMMCR_Mom_Vanilla_1"),
    layers=get_layers("r34_eMMCR_Mom_Vanilla_1"),
)
