from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['densenet121'] = lambda: ModelCommitment(
    identifier='densenet121', activations_model=get_model("densenet121"), layers=get_layers("densenet121"))

model_registry['densenet161'] = lambda: ModelCommitment(
    identifier='densenet161', activations_model=get_model("densenet161"), layers=get_layers("densenet161"))

model_registry['densenet169'] = lambda: ModelCommitment(
    identifier='densenet169', activations_model=get_model("densenet169"), layers=get_layers("densenet169"))

model_registry['densenet201'] = lambda: ModelCommitment(
    identifier='densenet201', activations_model=get_model("densenet201"), layers=get_layers("densenet201"))
