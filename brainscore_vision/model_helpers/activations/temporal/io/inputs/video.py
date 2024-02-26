import numpy as np

class Video:
    """Video object that represents a video clip."""

    def __init__(self, path):
        import moviepy.editor as mpe
        self.source = path
        self._video = mpe.VideoFileClip(path)
        self._frames = None
        self.reset()

    def copy(self):
        video = Video(self.source)
        video._start = self._start
        video._end = self._end
        video._fps = self._fps
        video._size = self._size
        self._frames = self._frames
        return video
    
    def view(self):
        import moviepy.editor as mpe
        video = self._video.set_fps(self._fps).resize(self._size) 
        
        # pad if start < 0 or end > duration
        start = self._start
        end = self._end
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
        
        video =  video.subclip(start, end)
        return video
    
    def reset(self):
        self._start = 0.
        self._end = self._video.duration
        self._fps = self._video.fps
        self._size = tuple(self._video.size)  # (height, width)
        self._frames = None
        return self

    @property
    def duration(self):
        return self.view().duration
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def num_frames(self):
        c = 0
        for _ in self.to_frames():
            c += 1
        return c
    
    @property
    def frame_size(self):
        return self._size
    
    @property
    def start_time(self):
        return self._start
    
    @property
    def end_time(self):
        return self._end
    
    ### Transformations
    
    def set_fps(self, fps):
        self._fps = fps
        self._frames = None
        return self
    
    def set_size(self, size):
        # size: (width, height)
        self._size = tuple(size)
        self._frames = None
        return self

    def set_window(self, start, end):
        # use ms as the time scale
        if end < start:
            raise ValueError("end time is earlier than start time")
        if end > self._video.duration:
            raise ValueError("end time is out of range")
        self._start = start
        self._end = end
        self._frames = None
        return self
    
    ### I/O
    def from_path(path):
        return Video(path)

    def to_path(self):
        # use context manager ?
        path = None  # make a temporal file
        raise NotImplementedError()
        return path
    
    def to_frames(self):
        if self._frames is None:
            self._frames = [frame for frame in self.view().iter_frames()]
        return self._frames

    def to_numpy(self):
        ret = []
        for frame in self.to_frames():
            ret.append(frame)
        ret = np.array(ret)
        return ret
    
    def to_pil_imgs(self):
        from PIL import Image
        return [Image.fromarray(frame) for frame in self.to_numpy()]
    
    def _properties(self):
        return (
            self.source,
            self.fps,
            self.frame_size,
            self.start_time,
            self.end_time,
        )
    
    def __eq__(self, other):
        for p1, p2 in zip(self._properties(), other._properties()):
            if isinstance(p1, float) or isinstance(p2, float):
                if not np.isclose(p1, p2):
                    return False
            else:
                if p1 != p2:
                    return False
        return True
    
    def __repr__(self):
        return f"{self.source}[[{self.start_time}->{self.end_time}]]"
    
    def __hash__(self):
        return hash(self._properties())
    

if __name__ == "__main__":
    video1 = Video.from_path("test.mp4")
    video2 = Video.from_path("test.mp4")
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()
    
    video2.set_fps(1)
    assert not (video1 == video2)

    video2.set_fps(video1.fps)
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()

    video2.set_window(0, 1000)
    assert not (video1 == video2)

    video2.reset()
    video2.set_window(0, 2000)
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()

    video2.set_window(10, 100)
    video1.set_window(10, 100)
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()

    video1.reset()
    video2.reset()
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()

    video2.set_size((100, 100))
    assert not (video1 == video2)

    video2.reset()
    assert (video1 == video2)
    assert (video1.to_numpy() == video2.to_numpy()).all()

    video2.set_fps(1)
    video3 = video2.copy()
    video3.reset()
    assert (video3 == video1)
    assert (video1.to_numpy() == video3.to_numpy()).all()
    assert not (video1 == video2)