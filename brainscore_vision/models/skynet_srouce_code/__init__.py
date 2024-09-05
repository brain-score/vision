from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['skynet_source_code'] = lambda: ModelCommitment(identifier='skynet_source_code', activations_model=get_model('skynet_source_code'), layers=get_layers('skynet_source_code'))
