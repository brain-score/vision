
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import sys 
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model, LAYERS



model_registry['blt_vs'] = lambda: ModelCommitment(
    identifier='blt_vs',
    activations_model=get_model(),
    layers=LAYERS)

