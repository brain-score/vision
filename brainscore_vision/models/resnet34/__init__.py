from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['resnet34'] = lambda: ModelCommitment(identifier='resnet34',
                                                               activations_model=get_model(),
                                                               layers=get_layers())