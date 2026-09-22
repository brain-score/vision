from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['jem_imagenet_p0p5_s1'] = lambda: ModelCommitment(identifier='jem_imagenet_p0p5_s1', activations_model=get_model('jem_imagenet_p0p5_s1'), layers=get_layers('jem_imagenet_p0p5_s1'), visual_degrees=8)
