import numpy as np
import os
import pytest

from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.core.video import TemporalInferencer, CausalInferencer 
from brainscore_vision.model_helpers.activations.temporal.inputs import Video

import logging
logging.basicConfig(level=logging.DEBUG)


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def get_transform_video(transform_img):
    def transform_video(video):
        import torch
        frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = transform_img(frames)
        return frames.permute(1, 0, 2, 3)
    return transform_video


def get_resnet_base_models(identifier, causal=False, **kwargs):
    from torchvision import transforms

    img_transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    vid_transform = get_transform_video(img_transform)

    if identifier == "R3D":
        activations_spec = {
                "stem": "CTHW",
                **{f'layer{i}': "CTHW" for i in range(1, 5)},
                "avgpool": "C",
                "fc": "C"
            }
        model_name = "r3d_18"
        def process_output(layer, layer_name, input, output):
            if layer_name == "avgpool":
                return output[:, :, 0, 0, 0]
            else:
                return output

    from torchvision.models import video as vid
    model = getattr(vid, model_name)(weights=None)

    inferencer_cls = TemporalInferencer if not causal else CausalInferencer
    wrapper = PytorchWrapper(identifier, model, vid_transform, process_output=process_output, 
                             inferencer_cls=inferencer_cls, fps=25, activations_spec=activations_spec, **kwargs)
    layers = list(activations_spec.keys())
    return wrapper, layers


def test_video():
    video1 = Video.from_path(os.path.join(os.path.dirname(__file__), "dots1.mp4"))
    video2 = Video.from_img(os.path.join(os.path.dirname(__file__), "../activations/rgb.jpg"), 1000, 30)

    assert video2.duration == 1000

    video3 = video1.set_window(-10, 0, padding="repeat")
    video4 = video1.set_window(-20, -10, padding="repeat")
    assert (video3.to_numpy() == video4.to_numpy()).all()

    assert video2.fps == 30
    assert video2.set_fps(1).to_numpy().shape[0] == 1

    video5 = video1.set_size((100, 100))
    assert video5.to_numpy().shape[1] == 100

    video1 = video1.set_fps(60)
    # assert Video.concat(video2, video1).fps == Video.concat(video1, video2).fps
    # assert Video.concat(video2, video1).fps == video1.fps

    for v in [video1, video2]:
        target_num_frames = 7
        duration = 1000 / v.fps * target_num_frames
        common = list(np.arange(0, v.duration, 100))
        extra1 = 1000 / v.fps * 3 + v.duration
        extra2 = 1000 / v.fps * 2 + v.duration
        extra3 = 1000 / v.fps * 1
        for t in [extra1, extra2, extra3] + common:
            video = v.set_window(t-duration, t, padding="repeat")
            assert video.to_numpy().shape[0] == target_num_frames

def test_video_load_frames():
    video1 = Video.from_path(os.path.join(os.path.dirname(__file__), "dots1.mp4"))
    fps = video1.fps
    duration = video1.duration
    all_frames = video1.to_frames()
    interval = 1000 / fps
    for t, f in zip(np.arange(interval, duration, interval), all_frames):
        v = video1.set_window(0, t)
        this_f = v.to_frames()[-1]
        assert (this_f == f).all()


@pytest.mark.memory_intense
@pytest.mark.parametrize("model_name", ["R3D"])
@pytest.mark.parametrize(["causal", "padding", "time_align"], [(False, True, "ignore_time"), (True, False, "evenly_spaced")])
def test_from_video_path(model_name, causal, padding, time_align):
    video_names = ["dots1.mp4", "dots2.mp4"]  # 2s vs 6s
    stimuli_paths = [os.path.join(os.path.dirname(__file__), video_name) for video_name in video_names]

    activations_extractor, layers = get_resnet_base_models(model_name, causal=causal, 
                                                           batch_padding=padding, time_alignment=time_align)
    activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                   layers=layers)

    assert activations is not None
    assert len(activations['stimulus_path']) == 2
    assert len(np.unique(activations['layer'])) == len(layers)

    expected_num_time_bins = 6 * 25
    if causal:
        assert activations.sizes['time_bin'] == expected_num_time_bins 

    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point
    return activations


def _build_stimulus_set(video_names):
    stimulus_set = StimulusSet([{'stimulus_id': video_name, 'some_meta': video_name[::-1]}
                                for video_name in video_names])
    stimulus_set.stimulus_paths = {video_name: os.path.join(os.path.dirname(__file__), video_name)
                                   for video_name in video_names}
    return stimulus_set

@pytest.mark.memory_intense
@pytest.mark.parametrize("model_name", ["R3D"])
@pytest.mark.parametrize(["causal", "padding"], [(False, True), (True, False)])
def test_from_stimulus_set(model_name, causal, padding):
    video_names = ["dots1.mp4", "dots2.mp4"]
    stimulus_set = _build_stimulus_set(video_names)

    activations_extractor, layers = get_resnet_base_models(model_name, causal=causal, batch_padding=padding)
    activations = activations_extractor(stimulus_set, layers=layers)
    
    assert activations is not None
    assert set(activations['stimulus_id'].values) == set(video_names)
    assert all(activations['some_meta'].values == [video_name[::-1] for video_name in video_names])
    assert len(np.unique(activations['layer'])) == len(layers)

    import gc
    gc.collect()  # free some memory, we're piling up a lot of activations at this point


test_from_stimulus_set("R3D", False, False)