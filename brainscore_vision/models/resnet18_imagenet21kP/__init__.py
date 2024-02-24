from brainscore_vision import model_registry
from .model import get_model

MODEL_NAME = "resnet18_imagenet21kP"

model_registry["resnet18_imagenet21kP-abdulkadir.gokce@epfl.ch"] = lambda: get_model(
    MODEL_NAME
)
