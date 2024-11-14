from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["mvimgnet_ss_00"] = lambda: ModelCommitment(
    identifier="mvimgnet_ss_00",
    activations_model=get_model("mvimgnet_ss_00"),
    layers=get_layers("mvimgnet_ss_00"),
)
