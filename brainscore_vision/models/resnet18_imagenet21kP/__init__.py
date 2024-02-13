from brainscore_vision import model_registry
from .model import get_model

MODEL_NAME = 'resnet18_imagenet21kP'
SUFFIX = "abdulkadir.gokce@epfl.ch"

model_registry[f'{MODEL_NAME}-{SUFFIX}'] = lambda: get_model(MODEL_NAME)
