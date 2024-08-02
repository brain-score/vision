from brainscore_vision import model_registry
from .model import get_model

model_registry["resnet18_imagenet21kP"] = lambda: get_model(
    "resnet18_imagenet21kP"
)
