from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['dinov2_vits14_linear'] = lambda: ModelCommitment(
    identifier='dinov2_vits14_linear', activations_model=get_model("dinov2_vits14_linear"), layers=get_layers("dinov2_vits14_linear"))

model_registry['dinov2_vitb14_linear'] = lambda: ModelCommitment(
    identifier='dinov2_vitb14_linear', activations_model=get_model("dinov2_vitb14_linear"), layers=get_layers("dinov2_vitb14_linear"))

model_registry['dinov2_vitl14_linear'] = lambda: ModelCommitment(
    identifier='dinov2_vitl14_linear', activations_model=get_model("dinov2_vitl14_linear"), layers=get_layers("dinov2_vitl14_linear"))

model_registry['dinov2_vitg14_linear'] = lambda: ModelCommitment(
    identifier='dinov2_vitg14_linear', activations_model=get_model("dinov2_vitg14_linear"), layers=get_layers("dinov2_vitg14_linear"))