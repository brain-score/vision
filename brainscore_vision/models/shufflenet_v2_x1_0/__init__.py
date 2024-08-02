from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['shufflenet_v2_x1_0'] = \
    lambda: ModelCommitment(identifier='shufflenet_v2_x1_0', activations_model=get_model('shufflenet_v2_x1_0'), layers=get_layers('shufflenet_v2_x1_0'))