import numpy as np
import os
import pytest

from collections import OrderedDict

from brainscore_vision.model_helpers.activations.temporal.core import Inferencer, TemporalInferencer, CausalInferencer, BlockInferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus


"""This module tests model_helpers.activations.temporal.core.inferencer

    Different inferencers are tested:
    - Inferencer: the basic inferencer that does not enforce any temporal context
    - TemporalInferencer: the inferencer that aligns the activations to the video time
    - CausalInferencer: the inferencer that ensures the activations are causal
    - BlockInferencer: the inferencer that divides the video into blocks and infer the activations for each block
"""


video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]
img_path = os.path.join(os.path.dirname(__file__), "../../activations/rgb.jpg")


def dummy_get_features(model_inputs, layers):
    feature = np.stack(model_inputs)
    B, F, H, W, C = feature.shape
    feature = feature.reshape(B, F, H//80, 80, W//80, 80, C).mean((3, 5))[..., :2]  # BFHWC=B,F,6,3,2
    batch_activation = OrderedDict({
        "layer1": feature,
        "layer2": feature[:, 0, 0, 0]
    })
    return batch_activation

def dummy_preprocess(video):
    feature = video.to_numpy()[:, 200:680, 200:440, :]
    return feature

def time_down_sample_preprocess(video):
    feature = video.to_numpy()[::2, 200:680, 200:440, :]
    return feature

dummy_layer_activation_format = {
    "layer1": "THWC",
    "layer2": "C",
}

dummy_layers = ["layer1", "layer2"]


@pytest.mark.memory_intense
@pytest.mark.parametrize("max_spatial_size", [None, 2, 4])
def test_inferencer(max_spatial_size):
    inferencer = Inferencer(dummy_get_features, dummy_preprocess, dummy_layer_activation_format, 
                            Video, max_workers=1, max_spatial_size=max_spatial_size, batch_grouper=lambda s: s.duration)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    if max_spatial_size is None:
        # 6 second video with fps 60 has 360 frames
        # the model simply return the same number of frames as the temporal size of activations
        # so the number of channel_temporal should be 360
        assert model_assembly.sizes["neuroid"] == 360*6*3*2 + 2
    else:
        assert model_assembly.sizes["neuroid"] == 360*max_spatial_size*(max_spatial_size//2) * 2 + 2
    assert model_assembly.sizes["stimulus_path"] == 2 


@pytest.mark.parametrize("time_alignment", ["evenly_spaced", "ignore_time"])
@pytest.mark.parametrize("fps", [10, 30, 45])
def test_temporal_inferencer(time_alignment, fps):
    inferencer = TemporalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, max_workers=1, 
                                    fps=fps, time_alignment=time_alignment)
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

    # manual computation check
    output_values = model_assembly.sel(stimulus_path=video_paths[1])\
                                    .isel(neuroid=model_assembly.layer=="layer1")\
                                    .transpose('time_bin', 'neuroid').values.reshape(-1)
    
    manual_compute_values = []
    video = Video.from_path(video_paths[1]).set_fps(fps)
    manual_compute_values = dummy_get_features([dummy_preprocess(video)], ["layer1"])["layer1"][0].reshape(-1)
    manual_compute_values = manual_compute_values.astype(output_values.dtype)
    assert np.allclose(output_values, manual_compute_values)


@pytest.mark.memory_intense
def test_img_input():
    fps = 30
    inferencer = TemporalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, max_workers=1, 
                                    fps=fps, convert_img_to_video=True, img_duration=1000)
    model_assembly = inferencer([img_path], layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == fps


def test_compute_temporal_context():
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


@pytest.mark.memory_intense
@pytest.mark.parametrize("preprocess", ["normal", "downsample"])
def test_causal_inferencer(preprocess):
    if preprocess == "normal":
        preprocess = dummy_preprocess
    else:
        preprocess = time_down_sample_preprocess
    fps = 10
    inferencer = CausalInferencer(dummy_get_features, dummy_preprocess, 
                                    dummy_layer_activation_format, 
                                    fps=fps, max_workers=1)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == 6 * fps
    assert np.isclose(model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0], 1000/fps)
    assert inferencer._compute_temporal_context() == (100, np.inf)

    # manual computation check
    output_values = model_assembly.sel(stimulus_path=video_paths[1])\
                                    .isel(neuroid=model_assembly.layer=="layer1")\
                                    .transpose('time_bin', 'neuroid').values
    manual_compute_values = []
    video = Video.from_path(video_paths[1]).set_fps(fps)
    interval = 1000/fps
    for time_end in np.arange(interval, 6000+interval, interval):
        clip = video.set_window(0, time_end)
        manual_compute_values.append(dummy_get_features([dummy_preprocess(clip)], ["layer1"])["layer1"][0, -1])
    manual_compute_values = np.stack(manual_compute_values).reshape(len(manual_compute_values), -1).astype(output_values.dtype)
    assert np.allclose(output_values, manual_compute_values)


@pytest.mark.memory_intense
@pytest.mark.parametrize("preprocess", ["normal", "downsample"])
def test_block_inferencer(preprocess):
    if preprocess == "normal":
        preprocessing = dummy_preprocess
    else:
        preprocessing = time_down_sample_preprocess
    fps = 10
    inferencer = BlockInferencer(dummy_get_features, preprocessing, dummy_layer_activation_format, fps=fps, 
                                 duration=(200, 4000), temporal_context_strategy="greedy", max_workers=1)
    model_assembly = inferencer(video_paths, layers=dummy_layers)
    assert model_assembly.sizes["time_bin"] == 8 * fps  # block overflow 2 x 4 seconds
    assert np.isclose(model_assembly['time_bin_end'].values[0] - model_assembly['time_bin_start'].values[0], 1000/fps)

    # manual computation check
    output_values = model_assembly.sel(stimulus_path=video_paths[1])\
                                    .isel(neuroid=model_assembly.layer=="layer1")\
                                    .transpose('time_bin', 'neuroid').values
    manual_compute_values = []
    video = Video.from_path(video_paths[1]).set_fps(fps)
    interval = 4000
    for time_end in np.arange(interval, 6000+interval, interval):
        time_start = time_end - interval
        clip = video.set_window(time_start, time_end)
        manual_compute_values.append(dummy_get_features([preprocessing(clip)], ["layer1"])["layer1"][0])
    manual_compute_values = np.concatenate(manual_compute_values)
    manual_compute_values = manual_compute_values.reshape(len(manual_compute_values), -1).astype(output_values.dtype)
    if preprocess == "downsample":
        output_values = output_values[::2]
    assert np.allclose(output_values, manual_compute_values)