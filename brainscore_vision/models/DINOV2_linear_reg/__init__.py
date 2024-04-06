from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['dinov2_vits14_reg_linear'] = lambda: ModelCommitment(
    identifier='dinov2_vits14_reg_linear', activations_model=get_model("dinov2_vits14_reg_linear"), layers=get_layers("dinov2_vits14_reg_linear"))