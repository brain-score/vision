from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_behavioral_readout_layer, get_layers, get_model


model_registry["deepeisnn_cifar10_vgg19_lif"] = lambda: ModelCommitment(
    identifier="deepeisnn_cifar10_vgg19_lif",
    activations_model=get_model("deepeisnn_cifar10_vgg19_lif"),
    layers=get_layers("deepeisnn_cifar10_vgg19_lif"),
    behavioral_readout_layer=get_behavioral_readout_layer("deepeisnn_cifar10_vgg19_lif"),
    visual_degrees=8,
)
