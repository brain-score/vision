import numpy as np
from typing import Union, Tuple, Callable, Hashable, List, Dict
from pathlib import Path

from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Stimulus
from brainscore_vision.model_helpers.activations.temporal.utils import assembly_align_to_fps, stack_with_nan_padding, data_assembly_mmap
from brainio.assemblies import NeuroidAssembly

from ..base import Inferencer


class TemporalInferencer(Inferencer):
    """Inferencer for video stimuli. The model takes video stimuli as input and generate the activations over time.
    Then, the activations will be aligned to video time by evenly distributing the activations of multiple time steps over time. The aligned
    activations will be again unified to the fps specified within the constructor (self.fps). Finally, the activations
    will be packaged into a NeuroidAssembly.
    
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

        if convert_img_to_video:
            assert img_duration is not None, "img_duration should be specified if convert_img_to_video is True"
        self.img_duration = img_duration
        self.convert_to_video = convert_img_to_video

    @property
    def identifier(self) -> str:
        id = f"{super().identifier}.fps={float(self.fps)}"
        if self.convert_to_video:
            id += f".img_dur={float(self.img_duration)}"
        return id

    def __call__(self, paths: List[Union[str, Path]], layers: List[str], mmap_path : str = None) -> NeuroidAssembly:
        stimuli = self.load_stimuli(paths)
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]
        num_time_bins = longest_stimulus.num_frames
        time_bin_coords = self._get_time_bin_coords(num_time_bins, self.fps)
        num_stimuli = len(paths)
        stimulus_paths = paths

        self._executor.add_stimuli(stimuli)
        data = None

        for temporal_layer_activations, indicies in self._executor.execute_batch(layers):
            for temporal_layer_activation, i in zip(temporal_layer_activations, indicies):
                stimulus = stimuli[i]
                # determine the time bin correspondence for each layer
                for t, layer_activation in self._disect_time(temporal_layer_activation, stimulus.num_frames):
                    if data is None:
                        num_feats, neuroid_coords = self._get_neuroid_coords(layer_activation, self._remove_T(self.layer_activation_format))
                        data = data_assembly_mmap(mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype, fill_value=np.nan)
                    flatten_activation = self._flatten_activations(layer_activation)
                    data[i, t, :] = flatten_activation

        data.register_meta(
            dims=["stimulus_path", "time_bin", "neuroid"],
            coords={
                "stimulus_path": stimulus_paths, 
                **neuroid_coords,
                **time_bin_coords,
            }, 
        )

        return data.to_assembly()

    def _disect_time(self, temporal_layer_activation, num_frames):
        paces = {}
        t_dims = {}
        for layer in temporal_layer_activation:
            activation = temporal_layer_activation[layer]
            specs = self.layer_activation_format[layer]
            t_dim = specs.index("T") if "T" in specs else None
            num_t = 1 if t_dim is None else activation.shape[t_dim]
            paces[layer] = num_t / num_frames  # evenly spaced activations over time
            t_dims[layer] = t_dim
        
        layer_ts = {layer: 0 for layer in temporal_layer_activation}
        for t in range(num_frames):
            ret = {}
            for layer in temporal_layer_activation:
                pace = paces[layer]
                t_dim = t_dims[layer]
                if t_dim is None:
                    ret[layer] = temporal_layer_activation[layer]
                else:
                    ret[layer] = temporal_layer_activation[layer].take(int(layer_ts[layer]), axis=t_dim)
                layer_ts[layer] += pace
            yield t, ret
    
    def _get_time_bin_coords(self, num_time_bins, fps):
        interval = 1000 / fps
        time_bin_starts = np.arange(0, num_time_bins) * interval
        time_bin_ends = time_bin_starts + interval
        return {
            "time_bin_start": ("time_bin", time_bin_starts),
            "time_bin_end": ("time_bin", time_bin_ends),
        }

    def _remove_T(self, layer_specs):
        ret = {}
        for layer, specs in layer_specs.items():
            ret[layer] = specs.replace("T", "")
        return ret

    def load_stimulus(self, path: Union[str, Path]) -> Video:
        # enable the conversion of images to videos
        if self.convert_to_video and Stimulus.is_image_path(path):
            video = Video.from_img_path(path, self.img_duration, self.fps)
        else:
            video = Video.from_path(path)
        video = video.set_fps(self.fps)
        self._check_video(video)
        return video

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
