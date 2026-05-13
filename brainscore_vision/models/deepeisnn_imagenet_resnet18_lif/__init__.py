from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_behavioral_readout_layer, get_layers, get_model


model_registry["deepeisnn_imagenet_resnet18_lif"] = lambda: ModelCommitment(
    identifier="deepeisnn_imagenet_resnet18_lif",
    activations_model=get_model("deepeisnn_imagenet_resnet18_lif"),
    layers=get_layers("deepeisnn_imagenet_resnet18_lif"),
    behavioral_readout_layer=get_behavioral_readout_layer("deepeisnn_imagenet_resnet18_lif"),
    visual_degrees=8,
)
