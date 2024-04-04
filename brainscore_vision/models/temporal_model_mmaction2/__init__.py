from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry["I3D"] = lambda: commit_model("I3D")
model_registry["I3D-nonlocal"] = lambda: commit_model("I3D-nonlocal")
model_registry["SlowFast"] = lambda: commit_model("SlowFast")
model_registry["X3D"] = lambda: commit_model("X3D")
model_registry["TimeSformer"] = lambda: commit_model("TimeSformer")
model_registry["VideoSwin-B"] = lambda: commit_model("VideoSwin-B")
model_registry["VideoSwin-L"] = lambda: commit_model("VideoSwin-L")
model_registry["UniFormer-V1"] = lambda: commit_model("UniFormer-V1")
model_registry["UniFormer-V2-B"] = lambda: commit_model("UniFormer-V2-B")
model_registry["UniFormer-V2-L"] = lambda: commit_model("UniFormer-V2-L")
