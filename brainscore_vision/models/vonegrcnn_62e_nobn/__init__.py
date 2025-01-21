from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vonegrcnn_62e_nobn'] = lambda: ModelCommitment(identifier='vonegrcnn_62e_nobn',
                                                               activations_model=get_model('vonegrcnn_62e_nobn'),
                                                               layers=get_layers('vonegrcnn_62e_nobn'))
