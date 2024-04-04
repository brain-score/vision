import numpy as np
import os
import pytest

from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.core import TemporalInferencer, CausalInferencer 


video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]
img_path = os.path.join(os.path.dirname(__file__), "../../activations/rgb.jpg")
fps = 1


def get_resnet_base_models(causal=False, **kwargs):
    from torchvision import transforms

    img_transform = transforms.Compose([
        transforms.Resize(4),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    def transform_video(video):
        import torch
        frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = img_transform(frames)
        return frames.permute(1, 0, 2, 3)
    
    from torch import nn
    class Dummy3DConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Conv3d(3, 2, kernel_size=(3, 3, 3), stride=(5, 5, 5), padding=(1, 1, 1))
            self.layer2 = nn.Conv3d(2, 2, kernel_size=(3, 3, 3), stride=(5, 5, 5), padding=(1, 1, 1))

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    layer_activation_format = {**{f'layer{i}': "CTHW" for i in range(1, 3)}}
    identifier = "dummy"
    model = Dummy3DConv()

    inferencer_cls = TemporalInferencer if not causal else CausalInferencer
    if inferencer_cls is CausalInferencer: 
        kwargs['duration'] = (0, 3000)
    wrapper = PytorchWrapper(identifier, model, transform_video, inferencer_cls=inferencer_cls, fps=fps, 
                             layer_activation_format=layer_activation_format, max_workers=1, batch_size=4, **kwargs)
    layers = list(layer_activation_format.keys())
    return wrapper, layers


@pytest.mark.memory_intense
@pytest.mark.parametrize(["causal", "padding", "time_align"], [(False, True, "ignore_time"), (True, False, "per_frame_aligned")])
def test_from_video_path(causal, padding, time_align):
    video_names = ["dots1.mp4", "dots2.mp4"]  # 2s vs 6s
    stimuli_paths = video_paths

    activations_extractor, layers = get_resnet_base_models(causal=causal, 
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
    return activations


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

    activations_extractor, layers = get_resnet_base_models(causal=causal, batch_padding=padding)
    activations = activations_extractor(stimulus_set, layers=layers)
    
    assert activations is not None
    assert set(activations['stimulus_id'].values) == set(video_names)
    assert all(activations['some_meta'].values == [video_name[::-1] for video_name in video_names])
    assert len(np.unique(activations['layer'])) == len(layers)

    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point