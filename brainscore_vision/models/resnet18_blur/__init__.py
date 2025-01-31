from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_blur'] = lambda: ModelCommitment(
    identifier='resnet18_blur',
    activations_model=get_model('resnet18_blur'),
    layers=get_layers('resnet18_blur')
)
