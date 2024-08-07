from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['mobilenet_v2_1-4_224_pytorch'] = lambda: ModelCommitment(identifier='mobilenet_v2_1-4_224_pytorch',
                                                               activations_model=get_model('mobilenet_v2_1-4_224_pytorch'),
                                                               layers=get_layers('mobilenet_v2_1-4_224_pytorch'))