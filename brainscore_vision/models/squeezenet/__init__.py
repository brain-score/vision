from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS


model_registry['squeezenet1_0'] = lambda: ModelCommitment(
    identifier='squeezenet1_0', activations_model=get_model("squeezenet1_0"), layers=LAYERS)

model_registry['squeezenet1_1'] = lambda: ModelCommitment(
    identifier='squeezenet1_1', activations_model=get_model("squeezenet1_1"), layers=LAYERS)
