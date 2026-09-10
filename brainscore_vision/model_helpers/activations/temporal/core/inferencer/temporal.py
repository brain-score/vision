from typing import Callable, Hashable, List, Tuple, Union
from pathlib import Path

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

from .base import Inferencer
from ...inputs import Stimulus, Video
from ...utils import data_assembly_mmap


MS_ROUNDING_DIGITS = 3


class TemporalInferencer(Inferencer):
    """Inferencer for video stimuli. The model takes a WHOLE video stimuli as input and generate the activations over time.
    Then, the activations will be aligned to video time by EVENLY distributing the activations of multiple time steps
    over time. The aligned activations will be again unified to the fps specified within the constructor (self.fps).
    Finally, the activations will be packaged into a NeuroidAssembly.

    Example:
        temporal_inferencer = TemporalInferencer(..., fps=10)
        model_assembly = temporal_inferencer(video_paths[1000ms], layers)
        model_assembly.time_bins -> [(0, 100), (100, 200), ..., (900, 1000)]  # 1000ms, 10fps

    Parameters
    ----------
    fps: float
        frame rate of the model sampling.

    num_frames: int, or (int, int)
        - If None, the model accepts videos of any length.
        - If a single int is passed, specify how many frames the model takes.
        - If a tuple of two ints is passed, specify the range of the number of frames the model takes (inclusive). If
          you need to specify infinite, use np.inf.

    duration: float, or (float, float)
        - If None, the model accepts videos of any length.
        - If a single float is passed, specify the duration of the model takes, in ms.
        - If a tuple of two floats is passed, specify the range of the duration the model takes (inclusive). If you need
          to specify infinite, use np.inf.

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
        fps: float,
        num_frames: Union[int, Tuple[int, int]] = None,
        duration: Union[float, Tuple[float, float]] = None,
        convert_img_to_video: bool = True,
        img_duration: float = 1000.0,
        batch_size: int = 32,
        batch_grouper: Callable[[Video], Hashable] = lambda video: (round(video.duration, 6), video.fps),
        **kwargs,
    ):
        super().__init__(*args, stimulus_type=Video, batch_size=batch_size, batch_grouper=batch_grouper, **kwargs)
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

    def __call__(self, paths: List[Union[str, Path]], layers: List[str], mmap_path: str = None) -> NeuroidAssembly:
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
                        num_feats, neuroid_coords = self._get_neuroid_coords(
                            layer_activation, self._remove_T(self.layer_activation_format)
                        )
                        data = data_assembly_mmap(
                            mmap_path,
                            shape=(num_stimuli, num_time_bins, num_feats),
                            dtype=self.dtype,
                            fill_value=np.nan,
                        )
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
            return (1 if type == "num_frames" else 0, np.inf)
        if isinstance(num, (tuple, list)):
            return num
        else:
            return (num, num)

    def _check_video(self, video: Video):
        if self.num_frames is not None:
            estimated_num_frames = int(self.fps * video.duration / 1000)
            assert (
                self.num_frames[0] <= estimated_num_frames <= self.num_frames[1]
            ), f"The number of frames must be within {self.num_frames}, but got {estimated_num_frames}"
        if self.duration is not None:
            assert (
                self.duration[0] <= video.duration <= self.duration[1]
            ), f"The duration must be within {self.duration}, but got {video.duration}"


class TemporalContextInferencerBase(TemporalInferencer):
    """Base class for context-aware inferencers (e.g., causal or block).

    It computes the allowable temporal context from `num_frames` and `duration`,
    then resolves an effective context length using `temporal_context_strategy`.

    Parameters
    ----------
    temporal_context_strategy: str
        How to pick the context length:
        - "greedy": use the maximum allowed context by the model.
        - "conservative": use the minimum allowed context by the model.
        - "fix": use `fixed_temporal_context`.
    fixed_temporal_context: float
        Fixed context length in ms (used only when strategy is "fix").
    out_of_bound_strategy: str
        How to pad when context runs past video bounds. Currently:
        - "repeat": repeat boundary frames.
    """

    def __init__(
        self,
        *args,
        temporal_context_strategy: str = "greedy",
        fixed_temporal_context: float = None,  # if None, default to greedy
        out_of_bound_strategy: str = "repeat",
        **kwargs,
    ):
        self.temporal_context_strategy = temporal_context_strategy
        self.fixed_temporal_context = fixed_temporal_context
        self.out_of_bound_strategy = out_of_bound_strategy
        if self.temporal_context_strategy == "fix" and self.fixed_temporal_context is None:
            raise ValueError("fixed_temporal_context must be specified if temporal_context_strategy is 'fix'.")
        super().__init__(*args, **kwargs)

    @property
    def identifier(self) -> str:
        lower, context = self._compute_temporal_context()
        lower = round(lower, MS_ROUNDING_DIGITS)
        context = round(context, MS_ROUNDING_DIGITS)
        to_add = f".context={context}>{lower}.oob={self.out_of_bound_strategy}"
        return f"{super().identifier}{to_add}"

    def load_stimulus(self, path):
        if self.convert_to_video and Stimulus.is_image_path(path):
            video = Video.from_img_path(path, self.img_duration, self.fps)
        else:
            video = Video.from_path(path)
        video = video.set_fps(self.fps)
        # does no check here
        return video

    def _overlapped_range(self, start1, end1, start2, end2):
        # compute the overlapped range of two ranges (start1, end1) and (start2, end2)
        lower, upper = max(start1, start2), min(end1, end2)
        if lower > upper:
            raise ValueError(f"Ranges [{start1}, {end1}] and [{start2}, {end2}] do not overlap.")
        return lower, upper

    def _compute_temporal_context(self):
        duration = self.duration
        num_frames = self.num_frames
        strategy = self.temporal_context_strategy

        interval = 1000 / self.fps
        num_frames_implied_ran = (num_frames[0] * interval, num_frames[1] * interval)
        ran = self._overlapped_range(*num_frames_implied_ran, *duration)
        lower = ran[0]

        if strategy in ["greedy", "conservative"]:
            if strategy == "greedy":
                return lower, ran[1]
            elif strategy == "conservative":
                return lower, ran[0]

        elif strategy == "fix":
            context = self.fixed_temporal_context if self.fixed_temporal_context is not None else ran[1]
            assert ran[0] <= context <= ran[1], f"Fixed temporal context {context} is not within the range {ran}"

        else:
            raise ValueError(f"Unknown temporal context strategy: {strategy}")

        return lower, context


class CausalInferencer(TemporalContextInferencerBase):
    """Causal inference over time: each output time uses only past frames.

    For each time bin, it feeds the model with a clip that ends at that time,
    then keeps only the last activation step. This yields a time series where
    each point is strictly causal.

    `inference_fps` controls the output time-bin spacing (defaults to model FPS).
    `fold_model_time` controls whether model time ("T") is folded into the neuroid axis
    (True) or reduced to the last time step (False, default).
    """

    def __init__(self, *args, inference_fps: float = None, fold_model_time: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_fps = self.fps if inference_fps is None else inference_fps
        self.fold_model_time = fold_model_time
        self.temporal_context_strategy = 'fix' if fold_model_time else self.temporal_context_strategy

    @property
    def identifier(self) -> str:
        base_id = super().identifier
        if self.fold_model_time:
            base_id = f"{base_id}.fold_model_time"
        if self.inference_fps == self.fps:
            return base_id
        return f"{base_id}.ifps={float(self.inference_fps)}"

    def __call__(self, paths, layers, mmap_path=None):
        lower, context = self._compute_temporal_context()
        interval = 1000 / self.inference_fps
        stimuli = self.load_stimuli(paths)
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]
        # Match the number of time steps implied by the per-stimulus arange loop.
        num_time_bins = int(np.ceil(longest_stimulus.duration / interval))
        num_stimuli = len(paths)
        time_bin_coords = self._get_time_bin_coords(num_time_bins, self.inference_fps)
        stimulus_paths = paths

        ts = []
        stimulus_index = []
        for s, stimulus in enumerate(stimuli):
            duration = stimulus.duration
            videos = []
            # here we ensure that the covered time range at least include the whole duration
            for t, time_end in enumerate(np.arange(interval, duration + interval, interval)):
                # see if the model only receive limited context
                time_start = self._get_time_start(time_end, context, lower)
                clip = stimulus.set_window(time_start, time_end, padding=self.out_of_bound_strategy)
                videos.append(clip)
                ts.append(t)
                stimulus_index.append(s)
            self._executor.add_stimuli(videos)

        data = None
        for temporal_layer_activations, indicies in self._executor.execute_batch(layers):
            for temporal_layer_activation, i in zip(temporal_layer_activations, indicies):
                s = stimulus_index[i]
                if self.fold_model_time:
                    layer_activation = temporal_layer_activation
                    layer_specs = self.layer_activation_format
                else:
                    layer_activation = self._get_last_time(temporal_layer_activation)
                    layer_specs = self._remove_T(self.layer_activation_format)
                if data is None:
                    num_feats, neuroid_coords = self._get_neuroid_coords(
                        layer_activation, layer_specs
                    )
                    data = data_assembly_mmap(
                        mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype, fill_value=np.nan
                    )
                flatten_activation = self._flatten_activations(layer_activation)
                t = ts[i]
                data[s, t, :] = flatten_activation

        data.register_meta(
            dims=["stimulus_path", "time_bin", "neuroid"],
            coords={
                "stimulus_path": stimulus_paths,
                **neuroid_coords,
                **time_bin_coords,
            },
        )

        return data.to_assembly()

    def _get_time_start(self, time_end, context, lower):
        assert context >= lower, f"Temporal context {context} is not within the range {lower}"
        if self.temporal_context_strategy == "fix":
            return time_end - context
        elif self.temporal_context_strategy == "greedy":
            proposed_time_start = time_end - context
            if proposed_time_start >= 0:
                return proposed_time_start
            else:
                if time_end < lower:
                    return time_end - lower
                else:
                    return 0
        elif self.temporal_context_strategy == "conservative":
            return time_end - context

    def _get_last_time(self, temporal_layer_activation):
        ret = {}
        for layer, activation in temporal_layer_activation.items():
            specs = self.layer_activation_format[layer]
            t_dim = specs.index("T") if "T" in specs else None
            ret[layer] = activation.take(-1, axis=t_dim) if t_dim is not None else activation
        return ret


class BlockInferencer(TemporalContextInferencerBase):
    """Block-wise inference: split a video into context-sized chunks.

    Each block is inferred separately and then stitched along time.
    Block size follows the resolved temporal context (from strategy, num_frames, duration).
    """

    def __call__(self, paths, layers, mmap_path=None):
        _, context = self._compute_temporal_context()

        if np.isinf(context):
            return super().__call__(paths, layers, mmap_path)

        EPS = 1e-6
        stimuli = self.load_stimuli(paths)
        num_stimuli = len(paths)
        stimulus_paths = paths

        num_blocks = []
        num_time_bins = []
        num_frames_per_block = None
        t_offsets = []
        stimulus_index = []
        for s, stimulus in enumerate(stimuli):
            duration = stimulus.duration
            video_blocks = []
            # for each stimulus, divide it into block clips with the specified context
            # EPS makes sure the last block is not out-of-bound
            for block_id, time_start in enumerate(np.arange(0, duration + EPS, context)):
                time_end = time_start + context
                clip = stimulus.set_window(time_start, time_end, padding=self.out_of_bound_strategy)
                video_blocks.append(clip)
                stimulus_index.append(s)
                clip_num_samples = clip.set_fps(self.fps).num_frames
                if num_frames_per_block is None:
                    num_frames_per_block = clip_num_samples
                else:
                    assert num_frames_per_block == clip_num_samples, "All blocks must have the same number of frames."
                t_offsets.append(block_id * num_frames_per_block)
            self._executor.add_stimuli(video_blocks)
            num_blocks.append(len(video_blocks))
            num_time_bins.append(len(video_blocks) * num_frames_per_block)

        num_time_bins = max(num_time_bins)
        time_bin_coords = self._get_time_bin_coords(num_time_bins, self.fps)

        data = None
        for temporal_layer_activations, indicies in self._executor.execute_batch(layers):
            for temporal_layer_activation, i in zip(temporal_layer_activations, indicies):
                s = stimulus_index[i]
                # determine the time bin correspondence for each layer
                for t, layer_activation in self._disect_time(temporal_layer_activation, num_frames_per_block):
                    if data is None:
                        num_feats, neuroid_coords = self._get_neuroid_coords(
                            layer_activation, self._remove_T(self.layer_activation_format)
                        )
                        data = data_assembly_mmap(
                            mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype, fill_value=np.nan
                        )
                    flatten_activation = self._flatten_activations(layer_activation)
                    t = t_offsets[i] + t
                    data[s, t, :] = flatten_activation

        data.register_meta(
            dims=["stimulus_path", "time_bin", "neuroid"],
            coords={
                "stimulus_path": stimulus_paths,
                **neuroid_coords,
                **time_bin_coords,
            },
        )

        return data.to_assembly()
