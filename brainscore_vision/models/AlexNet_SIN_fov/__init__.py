from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

### AlexNet_SIN with different fields of view than 8 visual degrees (default) ###
# exploring 4, 12, 16 visual degrees
# code is adapted from the original submission, please credit the original authors when using these models:
# https://github.com/brain-score/vision/tree/master/brainscore_vision/models/AlexNet_SIN


model_registry['AlexNet_SIN_fov4'] = lambda: ModelCommitment(
    identifier='AlexNet_SIN_fov4',
    activations_model=get_model('AlexNet_SIN_fov4'),
    layers=LAYERS,
    visual_degrees=4)

model_registry['AlexNet_SIN_fov12'] = lambda: ModelCommitment(
    identifier='AlexNet_SIN_fov12',
    activations_model=get_model('AlexNet_SIN_fov12'),
    layers=LAYERS,
    visual_degrees=12)

model_registry['AlexNet_SIN_fov16'] = lambda: ModelCommitment(
    identifier='AlexNet_SIN_fov16',
    activations_model=get_model('AlexNet_SIN_fov16'),
    layers=LAYERS,
    visual_degrees=16)
