from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["repvgg_b3"] = lambda: ModelCommitment(
    identifier="repvgg_b3", activations_model=get_model(), layers=LAYERS
)
