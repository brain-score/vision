from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["effnetb0_retrain"] = lambda: ModelCommitment(
    identifier="effnetb0_retrain",
    activations_model=get_model(),
    layers=LAYERS,
)
