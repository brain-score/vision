from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import get_layers, get_model


MODEL_IDENTIFIER = "deepeisnn_ei_cifar10_resnet18"


model_registry[MODEL_IDENTIFIER] = lambda: ModelCommitment(
    identifier=MODEL_IDENTIFIER,
    activations_model=get_model(MODEL_IDENTIFIER),
    layers=get_layers(MODEL_IDENTIFIER),
    behavioral_readout_layer=get_layers(MODEL_IDENTIFIER)[-1],
    visual_degrees=8,
)
