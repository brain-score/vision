
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import sys 
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model, LAYERS

region_layer_map = {
    "V1": "V1_5",
    "V2": "V2_5",
    "V4": "V4_5",
    "IT": "LOC_5"
}

model_registry['blt_vs'] = lambda: ModelCommitment(
    identifier='blt_vs',
    activations_model=get_model(),
    layers=LAYERS,
    region_layer_map=region_layer_map,
    visual_degrees=5)

