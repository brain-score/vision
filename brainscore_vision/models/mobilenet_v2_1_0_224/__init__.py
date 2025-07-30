from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['mobilenet_v2_1_0_224'] = lambda: ModelCommitment(identifier='mobilenet_v2_1_0_224',
                                                               activations_model=get_model('mobilenet_v2_1_0_224'),
                                                               layers=get_layers('mobilenet_v2_1_0_224'))
