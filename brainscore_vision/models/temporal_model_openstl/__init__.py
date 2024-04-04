from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry["ConvLSTM"] = lambda: commit_model("ConvLSTM")
model_registry["PredRNN"] = lambda: commit_model("PredRNN")
model_registry["SimVP"] = lambda: commit_model("SimVP")
model_registry["TAU"] = lambda: commit_model("TAU")
model_registry["MIM"] = lambda: commit_model("MIM")
