from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['compact_resnet50_robust_V4'] = lambda: ModelCommitment(identifier='compact_resnet50_robust_V4', activations_model=get_model('compact_resnet50_robust_V4'), layers=get_layers('compact_resnet50_robust_V4'))
