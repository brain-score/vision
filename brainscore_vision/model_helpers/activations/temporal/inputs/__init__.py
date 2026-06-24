import os
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image as PILImage

from ..utils import batch_2d_resize


class Stimulus:
    def from_path(self, path):
        raise NotImplementedError("Choose a concrete Stimulus type to use.")

    @staticmethod
    def is_video_path(path: Union[str, Path]) -> bool:
        extension = path.split('.')[-1].lower()
        return extension in ['mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv', 'gif']

    @staticmethod
    def is_image_path(path: Union[str, Path]) -> bool:
        extension = path.split('.')[-1].lower()
        return extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']


class Image(Stimulus):
    def __init__(self, path: str, size: int):
        self._path = path
        self._size = size

    def copy(self):
        return Image(self._path, self._size)

    @property
    def size(self):
        return self._size

    def set_size(self, size):
        img = self.copy()
        img._size = size
        return img

    def from_path(path):
        return Image(path, get_image_size(path))

    def to_pil_img(self):
        return PILImage.fromarray(self.to_numpy())

    def get_frame(self):
        return np.array(PILImage.open(self._path).convert('RGB'))

    # return (H, W, C[RGB])
    def to_numpy(self):
        arr = self.get_frame()

        if arr.shape[:2][::-1] != self._size:
            arr = batch_2d_resize(arr[None, :], self._size, "bilinear")[0]

        return arr

    def store_to_path(self, path):
        self.to_img().save(path)
        return path


def get_image_size(path):
    with PILImage.open(path) as img:
        size = img.size
    return size


EPS = 1e-9


def get_video_stats(video_path):
    assert os.path.exists(video_path), f"Video file {video_path} does not exist."
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (width, height)
    duration = length / fps * 1000
    cap.release()
    return fps, duration, size


def get_image_stats(image_path):
    with PILImage.open(image_path) as img:
        return img.size
    return size


class Video(Stimulus):
    """Video object that represents a video clip."""

    def __init__(
        self,
        path: Union[str, Path],
        fps: float,
        start: float,
        end: float,
        size: Tuple[int, int],
    ):
        self._path = path
        self._fps = fps
        self._size = size
        self._start = start
        self._end = end
        self._original_fps = None
        self._original_duration = None
        self._original_size = None

    def __getattribute__(self, key):
        if key.startswith("_original_"):
            if super().__getattribute__(key) is None:
                self._original_fps, self._original_duration, self._original_size = get_video_stats(self._path)
        return super().__getattribute__(key)

    def copy(self):
        # return view
        video = self.__class__(self._path, self._fps, self._start, self._end, self._size)
        video._original_fps = self._original_fps
        video._original_duration = self._original_duration
        video._original_size = self._original_size
        return video

    @property
    def duration(self):
        # in ms
        return self._end - self._start

    @property
    def fps(self):
        return self._fps

    @property
    def num_frames(self):
        return int(self.duration * self.fps / 1000 + EPS)

    @property
    def original_num_frames(self):
        return int(self._original_duration * self._original_fps / 1000 + EPS)

    @property
    def frame_size(self):
        return self._size

    ### Transformations: return copy

    def set_fps(self, fps):
        assert 1000 / fps <= self.duration, f"fps {fps} is too low for duration {self.duration}ms."
        video = self.copy()
        video._fps = fps
        return video

    def set_size(self, size):
        # size: (width, height)
        video = self.copy()
        video._size = size
        return video

    def set_window(self, start, end, padding="repeat"):
        # use ms as the time scale
        if end < start:
            raise ValueError("end time is earlier than start time")

        if padding != "repeat":
            raise NotImplementedError()

        video = self.copy()
        video._start = self._start + start
        video._end = self._start + end
        return video

    def _check_indices_ascending(self, indices):
        if len(indices) == 0:
            return False
        if len(indices) == 1:
            return True
        for i in range(1, len(indices)):
            if indices[i] < indices[i - 1]:
                return False
        return True

    def _sanitize_frames(self, frames, tol=0.01):
        # check if the read frames are valid
        # if some last frames are invalid, just copy the last valid frame
        # default tolerance: 0.01 of the total duration
        num_invalid = sum([f is None for f in frames])
        if num_invalid == len(frames):
            raise ValueError("No valid frames.")
        for i in range(num_invalid):
            assert frames[-1 - i] is None, "Invalid frames are not at the end."
        if num_invalid > int(self._original_duration / 1000 * self._fps * tol):
            raise ValueError("Too many invalid frames.")
        if num_invalid > 0:
            for i in range(num_invalid):
                frames[-1 - i] = frames[-1 - num_invalid]
            print(f"Warning: last {num_invalid} frames are invalid.")
        return frames

    def get_frames(self, indices):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")

        def _read(cap):
            ret, frame = cap.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ascending read optimization
        frames = []
        if self._check_indices_ascending(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])  # Move to the first frame index
            frame_index = indices[0] - 1
            for target_index in indices:
                to_move = target_index - frame_index
                for _ in range(to_move):
                    frame = _read(cap)
                frames.append(frame)
                frame_index += to_move
        else:
            # random access
            for i, index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # Move to the frame index
                frames.append(_read(cap))

        cap.release()
        frames = self._sanitize_frames(frames)

        return np.array(frames)

    ### I/O
    def from_path(path):
        fps, end, size = get_video_stats(path)
        start = 0
        return Video(path, fps, start, end, size)

    def from_img_path(img_path, duration, fps):
        # duration in ms
        size = get_image_stats(img_path)
        return VideoFromImage(img_path, fps, 0, duration, size)

    def to_numpy(self):
        # get the time stamps of frame samples
        start_frame = self._start * self._original_fps / 1000
        end_frame = self._end * self._original_fps / 1000
        # avoid taking the last extra frame
        samples = np.arange(start_frame, end_frame - EPS, self._original_fps / self.fps)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        original_num_frames = int(self._original_duration * self._original_fps / 1000 - EPS)  # EPS to avoid last frame OOB error
        sample_indices = np.clip(sample_indices, 0, original_num_frames - 1)

        # actual sampling
        frames = self.get_frames(sample_indices)

        # resizing
        if self._size != (frames.shape[2], frames.shape[1]):
            frames = batch_2d_resize(frames, self._size, "bilinear")

        return frames

    def to_frames(self):
        return [f for f in self.to_numpy()]

    def to_pil_imgs(self):
        return [PILImage.fromarray(frame) for frame in self.to_numpy()]

    def to_path(self):
        # use context manager ?
        path = None  # make a temporal file
        raise NotImplementedError()
        return path

    def store_to_path(self, path):
        # pick format based on path filename
        if path.endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif path.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError("Unsupported video format.")

        out = cv2.VideoWriter(path, fourcc, self._fps, self._size)
        for frame in self.to_frames():
            out.write(frame[..., ::-1])  # to RGB
        out.release()
        return path


class VideoFromImage(Video):
    def get_frames(self, indices):
        data = Image.from_path(self._path).to_numpy()
        N = len(indices)
        ret = np.repeat(data[np.newaxis, ...], N, axis=0)
        return ret
