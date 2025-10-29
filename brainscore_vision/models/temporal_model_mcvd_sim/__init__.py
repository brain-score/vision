from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry["MCVD-EGO4D-SIM"] = lambda: commit_model("MCVD-EGO4D-SIM")
model_registry["MCVD-PHYS-SIM"] = lambda: commit_model("MCVD-PHYS-SIM")
