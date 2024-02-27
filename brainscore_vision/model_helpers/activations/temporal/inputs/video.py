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
        video = Video(self._clip)
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
        c = 0
        for _ in self.to_frames():
            c += 1
        return c
    
    @property
    def frame_size(self):
        return self._clip.size
    
    @property
    def start_time(self):
        return self._clip.start
    
    @property
    def end_time(self):
        return self._clip.end
    
    ### Transformations: return copy
    
    def set_fps(self, fps):
        return Video(self._clip.set_fps(fps))
    
    def set_size(self, size):
        # size: (width, height)
        return Video(self._clip.resize(size))

    def set_window(self, start, end, padding="repeat"):
        # use ms as the time scale
        start = start / 1000
        end = end / 1000

        if end < start:
            raise ValueError("end time is earlier than start time")
        
        # pad if start < 0 or end > duration
        video = self._clip
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
                    img = video.get_frame(video.duration)
                    to_pad = mpe.ImageClip(img, duration=to_pad)
                    video = mpe.concatenate_videoclips([video, to_pad])

            elif padding == "off":
                raise ValueError("start time is earlier than 0 or end time is later than duration")
            else:
                raise NotImplementedError()
        
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
        for frame in self._clip.iter_frames():
            yield frame

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
    

if __name__ == "__main__":
    video1 = Video.from_path("/home/ytang/workspace/proj-brainscore-temporal/test.mp4")
    video2 = Video.from_img("/home/ytang/workspace/modules/tmp/brain-score/tests/test_benchmark_helpers/rgb1.jpg", 1000, 30)

    assert video2.duration == 1000

    video3 = video1.set_window(-10, 0, padding="repeat")
    video4 = video1.set_window(-20, -10, padding="repeat")
    assert (video3.to_numpy() == video4.to_numpy()).all()

    assert video2.fps == 30
    assert video2.set_fps(1).to_numpy().shape[0] == 1

    video1 = video1.set_fps(60)
    assert Video.concat(video2, video1).fps == Video.concat(video1, video2).fps
    assert Video.concat(video2, video1).fps == video1.fps