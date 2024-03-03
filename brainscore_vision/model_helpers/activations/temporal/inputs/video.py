import numpy as np
import moviepy.editor as mpe
from PIL import Image
from typing import Union

from .base import Stimulus


class Video(Stimulus):
    """Video object that represents a video clip."""

    def __init__(self, video_clip: mpe.VideoClip):
        self._clip = video_clip

    def copy(self):
        video = Video(self._clip.copy())
        return video
    
    @property
    def duration(self):
        # in ms
        return self._clip.duration * 1000
    
    @property
    def fps(self):
        return self._clip.fps
    
    @property
    def num_frames(self):
        return len(self.to_frames())
    
    @property
    def frame_size(self):
        return tuple(self._clip.size)
    
    @property
    def start_time(self):
        return self._clip.start * 1000
    
    @property
    def end_time(self):
        return self._clip.end * 1000
    
    ### Transformations: return copy
    
    def set_fps(self, fps):
        return Video(self._clip.set_fps(fps))
    
    def set_size(self, size):
        # size: (width, height)
        return Video(self._clip.resize(size))

    def set_window(self, start, end, padding="repeat"):
        # use ms as the time scale
        EPS = 1e-9  # avoid the additional frame at the end
        start = start / 1000
        end = end / 1000 - EPS

        if end < start:
            raise ValueError("end time is earlier than start time")
        

        # pad if start < 0 or end > duration
        video = self._clip.copy()
        if start < 0 or end > video.duration:
            if padding == "repeat":
                if start < 0:
                    to_pad = -start
                    img = video.get_frame(0)
                    to_pad = mpe.ImageClip(img, duration=to_pad)
                    video = mpe.concatenate_videoclips([to_pad, video])
                    end = end - start
                    start = 0

                if end > video.duration:
                    to_pad = end - video.duration
                    safe_margin = 1/video.fps  # BUG: here moviepy sometimes fails if you do get_frame(video.duration). See test_moviepy in tests for verification of this code.
                    img = video.get_frame(video.duration - safe_margin)
                    to_pad = mpe.ImageClip(img, duration=to_pad)
                    video = mpe.concatenate_videoclips([video, to_pad])

            elif padding == "off":
                raise ValueError("start time is earlier than 0 or end time is later than duration")
            else:
                raise NotImplementedError()
            
            # CAUTION: prevent weird behavior from moviepy: https://github.com/Zulko/moviepy/blob/a002df34a1b974e73cbea02c2f436c94b81fbc39/moviepy/video/io/ffmpeg_reader.py#L132
        
        return Video(video.subclip(start, end))
    
    ### I/O
    def from_path(path):
        return Video(mpe.VideoFileClip(path))
    
    def from_img(img: Union[Image.Image, str], duration, fps):
        # duration in ms
        return Video(mpe.ImageClip(img, duration=duration/1000).set_fps(fps))
    
    def to_path(self):
        # use context manager ?
        path = None  # make a temporal file
        raise NotImplementedError()
        return path
    
    def to_frames(self):
        if not hasattr(self, "_frames"):
            self._frames = list(self._clip.iter_frames())
        return self._frames

    def to_numpy(self):
        ret = []
        for frame in self.to_frames():
            ret.append(frame)
        ret = np.array(ret)
        return ret
    
    def to_pil_imgs(self):
        return [Image.fromarray(frame) for frame in self.to_numpy()]
    
    @staticmethod
    def concat(a, b):
        # fps will take the higher one
        video_clip = mpe.concatenate_videoclips([a._clip, b._clip])
        return Video(video_clip)
    