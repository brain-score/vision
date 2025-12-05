from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['alexnet_fov4'] = lambda: ModelCommitment(
    identifier='alexnet_fov4',
    activations_model=get_model('alexnet_fov4'),
    layers=LAYERS,
    visual_degrees=4.0)

model_registry['alexnet_fov12'] = lambda: ModelCommitment(
    identifier='alexnet_fov12',
    activations_model=get_model('alexnet_fov12'),
    layers=LAYERS,
    visual_degrees=12.0)

model_registry['alexnet_fov16'] = lambda: ModelCommitment(
    identifier='alexnet_fov16',
    activations_model=get_model('alexnet_fov16'),
    layers=LAYERS,
    visual_degrees=16.0)
