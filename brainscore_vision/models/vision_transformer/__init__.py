from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['vit_b_16'] = lambda: ModelCommitment(
    identifier='vit_b_16', activations_model=get_model("vit_b_16"), layers=get_layers("vit_b_16"))

model_registry['vit_b_32'] = lambda: ModelCommitment(
    identifier='vit_b_32', activations_model=get_model("vit_b_32"), layers=get_layers("vit_b_32"))

model_registry['vit_l_16'] = lambda: ModelCommitment(
    identifier='vit_l_16', activations_model=get_model("vit_l_16"), layers=get_layers("vit_l_16"))

model_registry['vit_l_32'] = lambda: ModelCommitment(
    identifier='vit_l_32', activations_model=get_model("vit_l_32"), layers=get_layers("vit_l_32"))

