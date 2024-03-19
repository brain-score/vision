from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=get_specified_layers(activations_model)
    region_layer_map={BrainModel.RecordingTarget.whole_brain: layers}
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers, region_layer_map=region_layer_map)


model_registry["GDT-Kinetics400"] = lambda: commit_model("GDT-Kinetics400")
model_registry["GDT-HowTo100M"] = lambda: commit_model("GDT-HowTo100M")
model_registry["GDT-IG65M"] = lambda: commit_model("GDT-IG65M")
