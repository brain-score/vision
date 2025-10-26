import numpy as np
from collections import OrderedDict

from .base import TemporalContextInferencerBase
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding, data_assembly_mmap
from brainio.assemblies import NeuroidAssembly


class CausalInferencer(TemporalContextInferencerBase):
    """Inferencer that ensures the activations are causal by feeding in the video in a causal manner. 

    Specifically, suppose the video lasts for 1000ms and the model samples every 100ms.
    Then, the activations from the last time step of the activations from the following videos:
    [0 ~ 100ms], [0 ~ 200ms], ..., [0 ~ 1000ms].
    will be stacked together to form the final activations for the video. In this way, the 
    activations are always "causal", which means that the temporal contexts of the model are 
    always from the past, not the future.

    If num_frames or duration is given, the model's temporal context will be set to match the two.
    """
    def __init__(
            self, 
            *args,
            **kwargs
        ):
        if "time_alignment" in kwargs:
            if kwargs["time_alignment"] != "per_frame_aligned":
                raise ValueError("CausalInferencer enforces time_alignment='per_frame_aligned'.")
            else:
                del kwargs["time_alignment"]
        super().__init__(*args, **kwargs, time_alignment="per_frame_aligned")
    
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

    def __call__(self, paths, layers, mmap_path=None):
        lower, context = self._compute_temporal_context()
        interval = 1000 / self.fps
        stimuli = self.load_stimuli(paths)
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]
        num_time_bins = longest_stimulus.num_frames
        num_stimuli = len(paths)
        time_bin_coords = self._get_time_bin_coords(num_time_bins, self.fps)
        stimulus_paths = paths

        ts = []
        stimulus_index = []
        for s, stimulus in enumerate(stimuli):
            duration = stimulus.duration
            videos = []
            # here we ensure that the covered time range at least include the whole duration
            for t, time_end in enumerate(np.arange(interval, duration+interval, interval)):
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
                layer_activation = self._get_last_time(temporal_layer_activation)
                if data is None:
                    num_feats, neuroid_coords = self._get_neuroid_coords(layer_activation, self._remove_T(self.layer_activation_format))
                    data = data_assembly_mmap(mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype, fill_value=np.nan)
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

    def _get_last_time(self, temporal_layer_activation):
        ret = {}
        for layer, activation in temporal_layer_activation.items():
            specs = self.layer_activation_format[layer]
            t_dim = specs.index("T") if "T" in specs else None
            ret[layer] = activation.take(-1, axis=t_dim) if t_dim is not None else activation
        return ret