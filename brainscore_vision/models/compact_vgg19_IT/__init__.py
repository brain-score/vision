from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['compact_vgg19_IT'] = lambda: ModelCommitment(identifier='compact_vgg19_IT', activations_model=get_model('compact_vgg19_IT'), layers=get_layers('compact_vgg19_IT'))
