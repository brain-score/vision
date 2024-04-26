import cv2
from decord import VideoReader
import numpy as np
from PIL import Image as PILImage
from typing import Tuple, Union
from pathlib import Path

from .base import Stimulus
from .image import Image
from brainscore_vision.model_helpers.activations.temporal.utils import batch_2d_resize


EPS = 1e-9  

def get_video_stats(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
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
            size: Tuple[int, int]
        ):
        self._path = path
        self._fps = fps
        self._size = size
        self._start = start
        self._end = end
        self._original_fps = None
        self._original_duration = None
        self._original_size = None

    def _set_original_stats(self):
        self._original_fps, self._original_duration, self._original_size = get_video_stats(self._path)

    def copy(self):
        # return view
        video = self.__class__(self._path, self._fps, self._start, self._end, self._size)
        video._original_fps = self.original_fps
        video._original_duration = self.original_duration
        video._original_size = self.original_frame_size
        return video
    
    @property
    def duration(self):
        # in ms
        return self._end - self._start
    
    @property
    def original_duration(self):
        if self._original_duration is None:
            self._set_original_stats()
        return self._original_duration
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def original_fps(self):
        if self._original_fps is None:
            self._set_original_stats()
        return self._original_fps
    
    @property
    def num_frames(self):
        return int(self.duration * self.fps/1000 + EPS)
    
    @property
    def original_num_frames(self):
        return int(self.original_duration * self.original_fps/1000 + EPS)
    
    @property
    def frame_size(self):
        return self._size
    
    @property
    def original_frame_size(self):
        if self._original_size is None:
            self._set_original_stats()
        return self._original_size
    
    ### Transformations: return copy
    
    def set_fps(self, fps):
        assert 1000/fps <= self.duration, f"fps {fps} is too low for duration {self.duration}ms."
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
    
    def get_frames(self, indices):
        reader = VideoReader(self._path)
        frames = reader.get_batch(indices).asnumpy()
        del reader
        return frames

    ### I/O
    def from_path(path):
        path = path
        fps, end, size = get_video_stats(path)
        start = 0
        return Video(path, fps, start, end, size)
    
    def from_img_path(img_path, duration, fps):
        # duration in ms
        size = get_image_stats(img_path)
        return VideoFromImage(img_path, fps, 0, duration, size)
    
    def to_numpy(self):
        # get the time stamps of frame samples
        start_frame = self._start * self.original_fps / 1000
        end_frame = self._end * self.original_fps / 1000
        # avoid taking the last extra frame
        samples = np.arange(start_frame, end_frame - EPS, self.original_fps/self.fps)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        sample_indices = np.clip(sample_indices, 0, self.original_num_frames-1)

        # actual sampling
        frames = self.get_frames(sample_indices)

        # resizing
        if self.frame_size != (frames.shape[2], frames.shape[1]):
            frames = batch_2d_resize(frames, self.frame_size, "bilinear")

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
    

class VideoFromImage(Video):
    def get_frames(self, indices):
        data = Image.from_path(self._path).to_numpy()
        N = len(indices)
        ret = np.repeat(data[np.newaxis, ...], N, axis=0)
        return ret