import numpy as np
import moviepy.editor as mpe
from PIL import Image as PILImage
from typing import Union

from .base import Stimulus
from .image import Image


class Video(Stimulus):
    """Video object that represents a video clip."""

    def __init__(self, frames: np.array, fps: int):
        self._frames = frames  # [F, H, W, C]
        self._fps = fps
        self._size = (frames.shape[2], frames.shape[1])  # (width, height)
        self._start = 0
        self._end = frames.shape[0] / fps * 1000  # in ms

    def copy(self):
        # return view
        video = Video(self._frames, self._fps)
        video._size = self._size
        video._start = self._start
        video._end = self._end
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
    
    @property
    def start_time(self):
        return self._start
    
    @property
    def end_time(self):
        return self._end
    
    ### Transformations: return copy
    
    def set_fps(self, fps):
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
        EPS = 1e-9  # avoid the additional frame at the end
        end = end - EPS

        if end < start:
            raise ValueError("end time is earlier than start time")
        
        if padding != "repeat":
            raise NotImplementedError()
        
        video = self.copy()
        video._start = self._start + start
        video._end = self._start + end
        return video
    
    ### I/O
    def _from_mpe_clip(clip):
        frames = np.array(list(clip.iter_frames()))
        ret = Video(frames, clip.fps)
        clip.close()
        return ret

    def from_path(path):
        mov = mpe.VideoFileClip(path)
        return Video._from_mpe_clip(mov)
    
    def from_img(img: Union[PILImage.Image, str, Image], duration, fps):
        # duration in ms
        if isinstance(img, Image):
            img = img._file
        mov = mpe.ImageClip(img, duration=duration/1000).set_fps(fps)
        return Video._from_mpe_clip(mov)
    
    def to_numpy(self):
        # actual conversion
        frames = self._frames

        # get the time stamps of frame samples
        interval = 1000 / self._fps  # ms
        samples = np.arange(self._start / interval, self._end / interval)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        sample_indices = np.clip(sample_indices, 0, len(frames)-1)

        # actual sampling
        frames = frames[sample_indices]

        # resizing
        if self._size != (frames.shape[2], frames.shape[1]):
            frames = np.array([mpe.ImageClip(frame).resize(self._size).get_frame(0) for frame in frames])

        return frames
    
    def to_frames(self):
        return [f for f in self.to_numpy()]

    def to_pil_imgs(self):
        return [Image.fromarray(frame) for frame in self.to_numpy()]
    
    def to_path(self):
        # use context manager ?
        path = None  # make a temporal file
        raise NotImplementedError()
        return path
    