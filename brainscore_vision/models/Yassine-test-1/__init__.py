from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["Yassine-test-1"] = lambda: ModelCommitment(identifier="Yassine-test-1", activations_model=get_model("Yassine-test-1"), layers=get_layers("Yassine-test-1"))
