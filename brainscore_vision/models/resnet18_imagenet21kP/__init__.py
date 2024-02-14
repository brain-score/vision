from brainscore_vision import model_registry
from .model import get_model

MODEL_NAME = "resnet18_imagenet21kP"
SUFFIX = "abdulkadir.gokce@epfl.ch"
MODEL_ID = "{}-{}".format(MODEL_NAME, SUFFIX)

model_registry[MODEL_ID] = lambda: get_model(MODEL_NAME)
