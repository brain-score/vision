from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['bl_mini_ecoset_new'] = lambda: ModelCommitment(identifier='bl_mini_ecoset_new', activations_model=get_model('bl_mini_ecoset_new'), layers=get_layers('bl_mini_ecoset_new'))