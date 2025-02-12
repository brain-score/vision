from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['mobilenet_v2_1_0_128'] = \
    lambda: ModelCommitment(identifier='mobilenet_v2_1_0_128', activations_model=get_model('mobilenet_v2_1_0_128'), layers=get_layers('mobilenet_v2_1_0_128'))