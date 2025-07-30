from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['efficientnet_b0'] = lambda: ModelCommitment(identifier='efficientnet_b0',
                                                                    activations_model=get_model('efficientnet_b0'),
                                                                    layers=get_layers('efficientnet_b0'))