from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['deit3_base_patch16_224_fb_in1k'] = lambda: ModelCommitment(identifier='deit3_base_patch16_224_fb_in1k', activations_model=get_model('deit3_base_patch16_224_fb_in1k'), layers=get_layers('deit3_base_patch16_224_fb_in1k'))
