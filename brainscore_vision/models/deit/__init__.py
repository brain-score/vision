from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['deit'] = lambda: ModelCommitment(
    identifier='deit',
    activations_model=get_model(),
    layers=LAYERS,
)
