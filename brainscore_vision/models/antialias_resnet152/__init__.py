from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['antialias-resnet152'] = lambda: ModelCommitment(identifier='antialias-resnet152',
                                                               activations_model=get_model('antialias-resnet152'),
                                                               layers=get_layers('antialias-resnet152'))