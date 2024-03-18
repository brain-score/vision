import numpy as np
import os
import pytest

from collections import OrderedDict

from brainscore_vision.model_helpers.activations.temporal.core import Inferencer, TemporalInferencer, CausalInferencer, BlockInferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus


video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]
img_path = os.path.join(os.path.dirname(__file__), "../../activations/rgb.jpg")


def dummy_get_features(model_inputs, layers):
    batch_activation = OrderedDict(
        {layer: model_inputs for layer in layers}
    )
    return batch_activation

def dummy_preprocess(video):
    return np.random.rand(video.num_frames, 6, 3)

dummy_layer_activation_format = {
    "layer1": "THW",
    "layer2": "THW",
}

dummy_layers = ["layer1", "layer2"]


@pytest.mark.parametrize("max_spatial_size", [None, 2, 4])
def test_inferencer(max_spatial_size):

    inferencer = Inferencer(dummy_get_features, dummy_preprocess, dummy_layer_activation_format, 
                            Video, max_spatial_size=max_spatial_size)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    if max_spatial_size is None:
        # 6 second video with fps 60 has 360 frames
        # the model simply return the same number of frames as the temporal size of activations
        # so the number of channel_temporal should be 360
        assert model_assembly.sizes["neuroid"] == 360*6*3 * len(dummy_layers)
    else:
        assert model_assembly.sizes["neuroid"] == 360*max_spatial_size**2//2 * len(dummy_layers)
    assert model_assembly.sizes["stimulus_path"] == 2 


@pytest.mark.parametrize("time_alignment", ["evenly_spaced", "ignore_time"])
@pytest.mark.parametrize("fps", [10, 30, 45])
def test_temporal_inferencer(time_alignment, fps):
    inferencer = TemporalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, fps=fps, time_alignment=time_alignment)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    assert model_assembly['time_bin_start'].values[0] == 0
    assert model_assembly['time_bin_end'].values[-1] == max(video_durations)

    if time_alignment != "ignore_time":
        # since the longer video lasts for 6 seconds, and the temporal inferencer align all output assembly to have fps
        # specified when constructing the inferencer, the number of time bins should be 6*fps
        assert model_assembly.sizes["time_bin"] == 6 * fps
        assert np.isclose(model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0], 1000/fps)
    else:
        assert model_assembly.sizes["time_bin"] == 1
        assert model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0] == max(video_durations)

def test_img_input():
    fps = 30
    inferencer = TemporalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, fps=fps, convert_img_to_video=True, img_duration=1000)
    model_assembly = inferencer([img_path], layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == fps

def test_temporal_context():
    fps=10
    inferencer = CausalInferencer(None, None, None, fps=fps, duration=(200, 1000), temporal_context_strategy="greedy")
    assert inferencer._compute_temporal_context() == (200, 1000)

    inferencer = CausalInferencer(None, None, None, fps=fps, duration=(200, 1000), temporal_context_strategy="conservative")
    assert inferencer._compute_temporal_context() == (200, 200)

    inferencer = CausalInferencer(None, None, None, fps=fps, duration=(0, 1000), num_frames=(2, 5), temporal_context_strategy="greedy")
    assert inferencer._compute_temporal_context() == (200, 500)

    inferencer = CausalInferencer(None, None, None, fps=fps, duration=(0, 1000), num_frames=(2, 15), temporal_context_strategy="greedy")
    assert inferencer._compute_temporal_context() == (200, 1000)

    inferencer = CausalInferencer(None, None, None, fps=fps, duration=(0, 1000), num_frames=(2, 15), temporal_context_strategy="fix", fixed_temporal_context=500)
    assert inferencer._compute_temporal_context() == (200, 500)

def test_causal_inferencer():
    fps = 10
    inferencer = CausalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, fps=fps)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == 6 * fps
    assert np.isclose(model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0], 1000/fps)
    assert inferencer._compute_temporal_context() == (100, np.inf)

def test_block_inferencer():
    fps = 10
    inferencer = BlockInferencer(dummy_get_features, dummy_preprocess, dummy_layer_activation_format, fps=fps, 
                                 duration=(200, 4000), temporal_context_strategy="greedy")
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == 8 * fps  # block overflow 2 x 4 seconds
    assert np.isclose(model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0], 1000/fps)