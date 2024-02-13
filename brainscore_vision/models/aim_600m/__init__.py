from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

MODEL_NAME = 'aim_600m'
SUFFIX="abdulkadir.gokce@epfl.ch"

model_func = lambda: ModelCommitment(
    identifier=MODEL_NAME, 
    activations_model=get_model(MODEL_NAME), 
    layers=get_layers(MODEL_NAME)
    )

model_registry[f'{MODEL_NAME}-{SUFFIX}'] = model_func