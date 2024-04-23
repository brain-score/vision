from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eSimCLR_lmda_001_1"] = lambda: ModelCommitment(
    identifier="eSimCLR_lmda_001_1",
    activations_model=get_model("eSimCLR_lmda_001_1"),
    layers=get_layers("eSimCLR_lmda_001_1"),
)
