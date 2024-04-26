import os
import numpy as np

from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Image


"""This module tests model_helpers.activations.temporal.inputs

    Different inputs are tested:
    - Video
    - Image
    Specifically, the different transformations (set fps, set size, etc.) of the inputs are tested.
"""

video_paths = [
    os.path.join(os.path.dirname(__file__), "..", "dots1.mp4"),
    os.path.join(os.path.dirname(__file__), "..", "dots2.mp4"),
]
video_durations = [2000, 6000]
img_path = os.path.join(os.path.dirname(__file__), "../../activations/rgb.jpg")


def test_video_load_frames():
    video1 = Video.from_path(video_paths[0])
    fps = video1.fps
    duration = video1.duration
    all_frames = video1.to_frames()
    interval = 1000 / fps
    for t, f in zip(np.arange(interval, duration, interval), all_frames):
        v = video1.set_window(0, t)
        this_f = v.to_frames()[-1]
        assert (this_f == f).all()

def test_video():
    video1 = Video.from_path(video_paths[0])
    video2 = Video.from_img_path(img_path, 1000, 30)

    assert video2.duration == 1000

    video3 = video1.set_window(-10, 0, padding="repeat")
    video4 = video1.set_window(-20, -10, padding="repeat")
    assert (video3.to_numpy() == video4.to_numpy()).all()

    assert video2.fps == 30
    assert video2.set_fps(1).to_numpy().shape[0] == 1

    video5 = video1.set_size((120, 100))
    assert tuple(video5.to_numpy().shape[1:3]) == (100, 120)

    video6 = video1.set_fps(30)
    assert (video6.to_numpy()[1] == video1.to_numpy()[2]).all()
    assert (video6.to_numpy()[2] == video1.to_numpy()[4]).all()

    video6 = video1.set_fps(20)
    assert (video6.to_numpy()[1] == video1.to_numpy()[3]).all()
    assert (video6.to_numpy()[2] == video1.to_numpy()[6]).all()

    video7 = video1.set_window(-100, 100).set_window(100, 200)
    assert video7.duration == 100
    assert (video7.to_numpy() == video1.set_window(0, 100).to_numpy()).all()

    video8 = video1.set_window(300, 500).set_window(0, 100)
    assert video8.duration == 100
    assert (video8.to_numpy() == video1.set_window(300, 400).to_numpy()).all()

    # test copy
    video9 = video1.set_fps(30).copy()
    assert (video9.to_numpy()[1] == video1.to_numpy()[2]).all()
    assert (video9.to_numpy()[2] == video1.to_numpy()[4]).all()

    # test padding
    video10 = video1.set_window(1000, 1000+1000/video1.fps)
    assert video10.to_numpy().shape[0] == 1
    assert (video10.to_numpy()[0] == video1.to_numpy()[int(video1.fps)]).all()

    for fps in [7.5, 9, 1, 43, 1000/video1.duration, 1001/video1.duration]:
        video9 = video1.set_fps(fps)
        assert video9.to_numpy().shape[0] == np.ceil(video1.duration * fps / 1000)

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

def test_image():
    img = Image.from_path(img_path)
    assert img.set_size((10, 12)).to_numpy().shape[:2] == (12, 10)