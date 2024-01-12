from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import PixelModel

model_registry['pixels'] = lambda: ModelCommitment(
    identifier='pixels',
    activations_model=PixelModel(),
    layers=['pixels'])
