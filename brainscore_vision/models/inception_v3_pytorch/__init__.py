from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['inception_v3_pytorch'] = lambda: ModelCommitment(identifier='inception_v3_pytorch',
                                                               activations_model=get_model('inception_v3_pytorch'),
                                                               layers=get_layers('inception_v3_pytorch'))