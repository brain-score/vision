from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry['r3d_18'] = lambda: commit_model('r3d_18')
model_registry['r2plus1d_18'] = lambda: commit_model('r2plus1d_18')
model_registry['mc3_18'] = lambda: commit_model('mc3_18')
model_registry['s3d'] = lambda: commit_model('s3d')
model_registry['mvit_v1_b'] = lambda: commit_model('mvit_v1_b')
model_registry['mvit_v2_s'] = lambda: commit_model('mvit_v2_s')