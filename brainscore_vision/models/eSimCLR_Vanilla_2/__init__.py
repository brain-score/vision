from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eSimCLR_Vanilla_2"] = lambda: ModelCommitment(
    identifier="eSimCLR_Vanilla_2",
    activations_model=get_model("eSimCLR_Vanilla_2"),
    layers=get_layers("eSimCLR_Vanilla_2"),
)
