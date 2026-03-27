from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['first_dev_learning_simple2'] = lambda: ModelCommitment(identifier='first_dev_learning_simple2',
                                                              activations_model=get_model('first_dev_learning_simple2'),
                                                              layers=get_layers('first_dev_learning_simple2'))
