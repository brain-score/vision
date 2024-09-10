from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['mobilenet_v2_0_75_160'] = \
    lambda: ModelCommitment(identifier='mobilenet_v2_0_75_160', activations_model=get_model('mobilenet_v2_0_75_160'), layers=get_layers('mobilenet_v2_0_75_160'))