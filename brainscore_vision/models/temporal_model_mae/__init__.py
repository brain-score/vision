from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    r = {'V1': 'encoder', 'V2': 'encoder', 'V4': 'encoder', 'IT': 'encoder'}
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers, region_layer_map=r)


model_registry["MAE-HUGE-Temporal"] = lambda: commit_model("MAE-HUGE-Temporal")
model_registry["MAE-BASE-Temporal"] = lambda: commit_model("MAE-BASE-Temporal")
model_registry["MAE-LARGE-Temporal"] = lambda: commit_model("MAE-LARGE-Temporal")