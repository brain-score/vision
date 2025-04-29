from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['yolos_small'] = lambda: ModelCommitment(identifier='yolos_small',
                                                               activations_model=get_model('yolos_small'),
                                                               layers=get_layers('yolos_small'))