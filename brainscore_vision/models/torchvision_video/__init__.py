from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

def commit_model(model_name):
    activations_model=get_model()
    layers=activations_model.base_model.layers()
    return ModelCommitment(identifier=model_name, activations_model=activations_model, layers=layers)

model_registry['r3d_18'] = commit_model('r3d_18')