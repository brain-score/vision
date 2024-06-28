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


model_registry["R3M-ResNet50-Temporal"] = lambda: commit_model("R3M-ResNet50-Temporal")
model_registry["R3M-ResNet34-Temporal"] = lambda: commit_model("R3M-ResNet34-Temporal")
model_registry["R3M-ResNet18-Temporal"] = lambda: commit_model("R3M-ResNet18-Temporal")
