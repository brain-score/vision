import numpy as np
from typing import Union, Tuple, Callable, Hashable, List, Dict
from pathlib import Path

from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus
from brainscore_vision.model_helpers.activations.temporal.utils import assembly_align_to_fps, stack_with_nan_padding
from brainio.assemblies import NeuroidAssembly

from ..base import Inferencer
from . import time_aligner as time_aligners 


class TemporalInferencer(Inferencer):
    """Inferencer for video stimuli. The model takes video stimuli as input and generate the activations over time.
    Then, the activations will be aligned to video time by the time_aligner specified in the constructor. The aligned
    activations will be again unified to the fps specified within the constructor (self.fps). Finally, the activations
    will be packaged into a NeuroidAssembly.
    
    NOTE: for all the time_alignment method, the inference of time bins will only be done with the longest video, but ignore all other input videos.

    Example:
        temporal_inferencer = TemporalInferenver(..., fps=10)
        model_assembly = temporal_inferencer(video_paths[1000ms], layers)
        model_assembly.time_bins -> [(0, 100), (100, 200), ..., (900, 1000)]  # 1000ms, 10fps

    Parameters
    ----------
    fps: float
        frame rate of the model sampling.

    num_frames: int, or (int, int)
        - If None, the model accepts videos of any length.
        - If a single int is passed, specify how many frames the model takes. 
        - If a tuple of two ints is passed, specify the range of the number of frames the model takes (inclusive). If you need to specify infinite, use np.inf.
    
    duration: float, or (float, float)
        - If None, the model accepts videos of any length.
        - If a single float is passed, specify the duration of the model takes, in ms.
        - If a tuple of two floats is passed, specify the range of the duration the model takes (inclusive). If you need to specify infinite, use np.inf.
    
    time_alignment: str
        specify the method to align the activations in time.
        The options and specifications are in the time_aligners module. The current options are:
        - evenly_spaced: align the activations to have evenly spaced time bins across the whole video time span.
        - ignore_time: ignore the time information and make a single time bin of the entire video.
        - estimate_layer_fps: estimate the fps of the layer based on the video fps.
        - per_frame_aligned: align the activations to the video frames.

    convert_img_to_video: bool
        whether to convert the input images to videos.
    img_duration: float
        specify the duration of the images, in ms. This will work only if convert_img_to_video is True.
    batch_size: int
        number of stimuli to process in each batch.
    batch_grouper: function
        function that takes a stimulus and return the property based on which the stimuli can be grouped.
    """
    def __init__(
            self,
            *args,
            fps : float,
            num_frames : Union[int, Tuple[int, int]] = None,
            duration : Union[float, Tuple[float, float]] = None,
            time_alignment : str = "evenly_spaced",
            convert_img_to_video : bool = True,
            img_duration : float = 1000.0,
            batch_size : int = 32,
            batch_grouper : Callable[[Video], Hashable] = lambda video: (round(video.duration, 6), video.fps),  # not including video.frame_size because most preprocessors will change the frame size to be the same
            **kwargs,
    ):
        super().__init__(*args, stimulus_type=Video, batch_size=batch_size, 
                         batch_grouper=batch_grouper, **kwargs)
        self.fps = fps
        self.num_frames = self._make_range(num_frames, type="num_frames")
        self.duration = self._make_range(duration, type="duration")
        assert hasattr(time_aligners, time_alignment), f"Unknown time alignment method: {time_alignment}"
        self.time_aligner = getattr(time_aligners, time_alignment)

        if convert_img_to_video:
            assert img_duration is not None, "img_duration should be specified if convert_img_to_video is True"
        self.img_duration = img_duration
        self.convert_to_video = convert_img_to_video

    @property
    def identifier(self) -> str:
        id = f"{super().identifier}.{self.time_aligner.__name__}.fps={float(self.fps)}"
        if self.convert_to_video:
            id += f".img_dur={float(self.img_duration)}"
        return id

    def load_stimulus(self, path: Union[str, Path]) -> Video:
        if self.convert_to_video and Stimulus.is_image_path(path):
            video = Video.from_img_path(path, self.img_duration, self.fps)
        else:
            video = Video.from_path(path)
        video = video.set_fps(self.fps)
        self._check_video(video)
        return video
    
    def package_layer(
            self, 
            layer_activations : List[np.array],
            layer_spec : str, 
            stimuli : List[Stimulus]
        ):
        assert len(layer_activations) == len(stimuli)
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]
        ignore_time = self.time_aligner is time_aligners.ignore_time
        channels = self._map_dims(layer_spec)
        layer_activations = stack_with_nan_padding(layer_activations)
        assembly = self._package(layer_activations, ["stimulus_path"] + channels)
        # align to the longest stimulus
        assembly = self.time_aligner(assembly, longest_stimulus)
        if "channel_temporal" in channels and not ignore_time: 
            channels.remove("channel_temporal")
        assembly = self._stack_neuroid(assembly, channels)
        if not ignore_time:
            assembly = assembly_align_to_fps(assembly, self.fps)
        return assembly

    def _make_range(self, num, type="num_frames"):
        if num is None:
            return (1 if type=='num_frames' else 0, np.inf)
        if isinstance(num, (tuple, list)):
            return num
        else:
            return (num, num)

    def _check_video(self, video: Video):
        if self.num_frames is not None:
            estimated_num_frames = int(self.fps * video.duration / 1000)
            assert self.num_frames[0] <= estimated_num_frames <= self.num_frames[1], f"The number of frames must be within {self.num_frames}, but got {estimated_num_frames}"
        if self.duration is not None:
            assert self.duration[0] <= video.duration <= self.duration[1], f"The duration must be within {self.duration}, but got {video.duration}"