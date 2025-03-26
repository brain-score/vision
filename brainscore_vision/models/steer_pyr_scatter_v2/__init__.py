from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["steer_pyer_scatter_v2"] = lambda: ModelCommitment(
    identifier="steer_pyer_scatter_v2",
    activations_model=get_model("steer_pyer_scatter_v2"),
    layers=get_layers("steer_pyer_scatter_v2"),
)
