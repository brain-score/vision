from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["dorinet_cornet_rt"] = lambda: ModelCommitment(
    identifier="dorinet_cornet_rt",
    activations_model=get_model(),
    layers=LAYERS,
)
