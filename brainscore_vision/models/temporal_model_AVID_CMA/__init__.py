from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry["AVID-CMA-Kinetics400"] = lambda: commit_model("AVID-CMA-Kinetics400")
model_registry["AVID-CMA-Audioset"] = lambda: commit_model("AVID-CMA-Audioset")
model_registry["AVID-Kinetics400"] = lambda: commit_model("AVID-Kinetics400")
model_registry["AVID-Audioset"] = lambda: commit_model("AVID-Audioset")
