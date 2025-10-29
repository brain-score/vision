from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['hmax_v3_adj'] = lambda: ModelCommitment(identifier='hmax_v3_adj', activations_model=get_model('hmax_v3_adj'), layers=get_layers('hmax_v3_adj'))
