import numpy as np
import os
import pytest


from brainscore_vision.model_helpers.activations.temporal.core import Inferencer, TemporalInferencer, CausalInferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus


video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]

## Test inferencers

def dummy_get_features(model_inputs, layers):
    return model_inputs

def dummy_preprocess(video):
    return np.random.rand(4, 3, 2)

def test_inferencer():
    inferencer = Inferencer(dummy_get_features, dummy_preprocess, Video)
    layers = ["layer1", "layer2"]
    model_assembly = inferencer(video_paths, layers=layers)
    assert model_assembly.sizes["neuroid"] == 4*3*2 * len(layers)
    assert model_assembly.sizes["presentation"] == 2 
    assert model_assembly["time_bin_start"].values[0] == 0
    assert model_assembly["time_bin_end"].values[-1] == max(video_durations)


test_inferencer()