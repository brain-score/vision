
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
model_registry['vgg16_less_variation_iteration=1'] = lambda: ModelCommitment(identifier='vgg16_less_variation_iteration=1')
