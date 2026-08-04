from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model


model_registry["deepeisnn_ei_cifar10_resnet18"] = lambda: ModelCommitment(
    identifier="deepeisnn_ei_cifar10_resnet18",
    activations_model=get_model("deepeisnn_ei_cifar10_resnet18"),
    layers=get_layers("deepeisnn_ei_cifar10_resnet18"),
    behavioral_readout_layer=get_layers("deepeisnn_ei_cifar10_resnet18")[-1],
    visual_degrees=8,
)
