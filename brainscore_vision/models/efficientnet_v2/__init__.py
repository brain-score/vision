from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['efficientnet_v2_s'] = lambda: ModelCommitment(
    identifier='efficientnet_v2_s', activations_model=get_model("efficientnet_v2_s"), layers=get_layers("efficientnet_v2_s"))

model_registry['efficientnet_v2_m'] = lambda: ModelCommitment(
    identifier='efficientnet_v2_m', activations_model=get_model("efficientnet_v2_m"), layers=get_layers("efficientnet_v2_m"))

model_registry['efficientnet_v2_l'] = lambda: ModelCommitment(
    identifier='efficientnet_v2_l', activations_model=get_model("efficientnet_v2_l"), layers=get_layers("efficientnet_v2_l"))
