import numpy as np
import os
import pytest

from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations.temporal.inputs.base import Stimulus
from brainscore_vision.model_helpers.activations.temporal.model import ActivationWrapper
from brainscore_vision.model_helpers.activations.temporal.core import TemporalInferencer, CausalInferencer 
from collections import OrderedDict


video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]
img_path = os.path.join(os.path.dirname(__file__), "../../activations/rgb.jpg")
fps = 1


def get_fake_models(causal=False, **kwargs):
    def transform_video(video):
        frames = video.to_numpy()[:, :12, :12]
        return frames
    
    class FakeActivationWrapper(ActivationWrapper):
        def __init__(self, **kwargs):
            super().__init__("dummy", transform_video, **kwargs)

        def get_activations(self, inputs, layers):
            ret = OrderedDict()
            for layer in layers:
                ret[layer] = np.stack(inputs)
            return ret

    layer_activation_format = {**{f'layer{i}': "THWC" for i in range(1, 3)}}

    inferencer_cls = TemporalInferencer if not causal else CausalInferencer
    if inferencer_cls is CausalInferencer: 
        kwargs['duration'] = (0, 3000)
    wrapper = FakeActivationWrapper(inferencer_cls=inferencer_cls, fps=fps, 
                             layer_activation_format=layer_activation_format, max_workers=1, batch_size=4, **kwargs)
    layers = list(layer_activation_format.keys())
    return wrapper, layers


@pytest.mark.memory_intense
@pytest.mark.parametrize(["causal", "padding", "time_align"], [(False, True, "ignore_time"), (True, False, "per_frame_aligned")])
def test_from_video_path(causal, padding, time_align):
    video_names = ["dots1.mp4", "dots2.mp4"]  # 2s vs 6s
    stimuli_paths = video_paths

    activations_extractor, layers = get_fake_models(causal=causal, 
                                                           batch_padding=padding, time_alignment=time_align)
    activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                   layers=layers)

    assert activations is not None
    assert len(activations['stimulus_path']) == 2
    assert len(np.unique(activations['layer'])) == len(layers)

    expected_num_time_bins = 6 * fps
    if causal:
        assert activations.sizes['time_bin'] == expected_num_time_bins 

    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point


def _build_stimulus_set(video_names):
    stimulus_set = StimulusSet([{'stimulus_id': video_name, 'some_meta': video_name[::-1]}
                                for video_name in video_names])
    stimulus_set.stimulus_paths = {video_name: path
                                   for video_name, path in zip(video_names, video_paths)}
    return stimulus_set


@pytest.mark.memory_intense
@pytest.mark.parametrize(["causal", "padding"], [(False, True), (True, False)])
def test_from_stimulus_set(causal, padding):
    video_names = ["dots1.mp4", "dots2.mp4"]
    stimulus_set = _build_stimulus_set(video_names)

    activations_extractor, layers = get_fake_models(causal=causal, batch_padding=padding)
    activations = activations_extractor(stimulus_set, layers=layers)
    
    assert activations is not None
    assert set(activations['stimulus_id'].values) == set(video_names)
    assert all(activations['some_meta'].values == [video_name[::-1] for video_name in video_names])
    assert len(np.unique(activations['layer'])) == len(layers)

    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point