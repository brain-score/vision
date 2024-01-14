from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["dorinet_cornet_z_trained_25"] = lambda: ModelCommitment(
    identifier="dorinet_cornet_z_trained_25",
    activations_model=get_model(),
    layers=get_layers(),
)
