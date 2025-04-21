from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['yolos_tiny_snnHead'] = lambda: ModelCommitment(identifier='yolos_tiny_snnHead',
                                                               activations_model=get_model('yolos_tiny_snnHead'),
                                                               layers=get_layers('yolos_tiny_snnHead'))