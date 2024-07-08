from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['antialiased-rnext101_32x8d'] = lambda: ModelCommitment(identifier='antialiased-rnext101_32x8d',
                                                               activations_model=get_model('antialiased-rnext101_32x8d'),
                                                               layers=get_layers('antialiased-rnext101_32x8d'))