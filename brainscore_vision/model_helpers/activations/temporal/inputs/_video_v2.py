import cv2
from decord import VideoReader
import numpy as np
from PIL import Image as PILImage
from typing import Union
import multiprocessing


from .base import Stimulus
from .image import Image
from brainscore_vision.model_helpers.activations.temporal.core.utils import cv2_resize

lock = multiprocessing.Lock()


def opencv_get_size(video_path):
    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid.release()
    return width, height


class FakeVideoReaderForImage:
    def __init__(self, img: Image, duration, fps):
        self._data = img.to_numpy()
        self._duration = duration
        self._fps = fps

    def get_batch(self, indices):
        N = len(indices)
        ret = np.repeat(self._data[np.newaxis, ...], N, axis=0)
        return ret
    
    def get_avg_fps(self):
        return self._fps
    
    def __len__(self):
        return int(self._duration * self._fps / 1000)


class Video(Stimulus):
    """Video object that represents a video clip."""

    def __init__(self, reader, size):
        self._reader = reader  # VideoReader
        self._fps = reader.get_avg_fps()
        self._original_fps = self._fps
        self._start = 0
        self._size = size
        self._end = len(reader) / self._fps * 1000  # in ms

    def copy(self):
        # return view
        video = Video(self._reader, self._size)
        video._start = self._start
        video._end = self._end
        video._original_fps = self._original_fps
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
        return int(self.duration * self.fps/1000)
    
    @property
    def frame_size(self):
        return self._size
    
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
    
    ### I/O
    def from_path(path):
        reader = VideoReader(path)
        size = opencv_get_size(path)
        return Video(reader, size)
    
    def from_img(img: Union[PILImage.Image, str, Image], duration, fps):
        # duration in ms
        if isinstance(img, PILImage.Image):
            raise
        elif isinstance(img, str):
            img = Image.from_path(img)
        elif isinstance(img, Image):
            raise
        reader = FakeVideoReaderForImage(img, duration, fps)
        return Video(reader, img.size)
    
    def to_numpy(self):
        # get the time stamps of frame samples
        start_frame = self._start * self._original_fps / 1000
        end_frame = self._end * self._original_fps / 1000
        EPS = 1e-9  # avoid taking the last extra frame
        samples = np.arange(start_frame, end_frame - EPS, self._original_fps/self._fps)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        sample_indices = np.clip(sample_indices, 0, self.num_frames-1)

        # actual sampling
        lock.acquire()
        frames = self._reader.get_batch(sample_indices).asnumpy()
        lock.release()

        # resizing
        if self._size != (frames.shape[2], frames.shape[1]):
            frames = cv2_resize(frames, self._size, cv2.INTER_LINEAR)

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
    