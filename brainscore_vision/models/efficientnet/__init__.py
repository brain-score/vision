from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['efficientnet_b0'] = lambda: ModelCommitment(
    identifier='efficientnet_b0', activations_model=get_model("efficientnet_b0"), layers=get_layers("efficientnet_b0"))

model_registry['efficientnet_b1'] = lambda: ModelCommitment(
    identifier='efficientnet_b1', activations_model=get_model("efficientnet_b1"), layers=get_layers("efficientnet_b1"))

model_registry['efficientnet_b2'] = lambda: ModelCommitment(
    identifier='efficientnet_b2', activations_model=get_model("efficientnet_b2"), layers=get_layers("efficientnet_b2"))

model_registry['efficientnet_b3'] = lambda: ModelCommitment(
    identifier='efficientnet_b3', activations_model=get_model("efficientnet_b3"), layers=get_layers("efficientnet_b3"))

model_registry['efficientnet_b4'] = lambda: ModelCommitment(
    identifier='efficientnet_b4', activations_model=get_model("efficientnet_b4"), layers=get_layers("efficientnet_b4"))

model_registry['efficientnet_b5'] = lambda: ModelCommitment(
    identifier='efficientnet_b5', activations_model=get_model("efficientnet_b5"), layers=get_layers("efficientnet_b5"))

model_registry['efficientnet_b6'] = lambda: ModelCommitment(
    identifier='efficientnet_b6', activations_model=get_model("efficientnet_b6"), layers=get_layers("efficientnet_b6"))

model_registry['efficientnet_b7'] = lambda: ModelCommitment(
    identifier='efficientnet_b7', activations_model=get_model("efficientnet_b7"), layers=get_layers("efficientnet_b7"))
