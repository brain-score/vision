from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['convnextv2-tiny'] = lambda: ModelCommitment(identifier='convnextv2-tiny', activations_model=get_model('convnextv2-tiny'), layers=get_layers('convnextv2-tiny'))