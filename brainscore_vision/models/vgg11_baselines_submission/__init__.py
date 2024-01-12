from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["neg_samples_e2e"] = lambda: ModelCommitment(
    identifier="neg_samples_e2e",
    activations_model=get_model(),
    layers=LAYERS,
)
