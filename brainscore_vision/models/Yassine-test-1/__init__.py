from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["Yassine_test_1"] = lambda: ModelCommitment(identifier="Yassine_test_1", activations_model=get_model("Yassine_test_1"), layers=get_layers("Yassine_test_1"))
