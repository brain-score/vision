from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)

model_registry["ResNet152-Temporal"] = lambda: commit_model("ResNet152-Temporal")
model_registry["ResNet101-Temporal"] = lambda: commit_model("ResNet101-Temporal")
model_registry["ResNet50-Temporal"] = lambda: commit_model("ResNet50-Temporal")
model_registry["ResNet34-Temporal"] = lambda: commit_model("ResNet34-Temporal")
model_registry["ResNet18-Temporal"] = lambda: commit_model("ResNet18-Temporal")
