from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["dorinet_cornet_rt_trained_v1"] = lambda: ModelCommitment(
    identifier="dorinet_cornet_rt_trained_v1",
    activations_model=get_model(),
    layers=get_layers(),
)
