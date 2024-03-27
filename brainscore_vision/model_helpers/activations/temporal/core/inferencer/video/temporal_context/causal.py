import numpy as np
from collections import OrderedDict

from .base import TemporalContextInferencerBase
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding


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

    def inference(self, stimuli, layers):
        interval = 1000 / self.fps
        lower, context = self._compute_temporal_context()
        num_clips = []
        latest_time_end = 0
        for inp in stimuli:
            duration = inp.duration
            videos = []
            # here we ensure that the covered time range at least include the whole duration
            for time_end in np.arange(interval, duration+interval, interval):
                # see if the model only receive limited context
                time_start = self._get_time_start(time_end, context, lower)
                clip = inp.set_window(time_start, time_end, padding=self.out_of_bound_strategy)
                latest_time_end = max(time_end, latest_time_end)
                videos.append(clip)

            self._executor.add_stimuli(videos)
            num_clips.append(len(videos))

        activations = self._executor.execute(layers)
        layer_activations = OrderedDict()
        for layer in layers:
            activation_dims = self.layer_activation_format[layer]
            clip_start = 0
            for num_clip in num_clips:
                video_activations = activations[layer][clip_start:clip_start+num_clip]  # clips for this video
                # make T the first dimension, as [T, ...]
                if 'T' in activation_dims:
                    time_index = activation_dims.index('T')
                    video_activations = [a.take(-1, axis=time_index) for a in video_activations]
                layer_activations.setdefault(layer, []).append(np.stack(video_activations, axis=0))
                clip_start += num_clip

        for layer in layers:
            layer_activations[layer] = stack_with_nan_padding(layer_activations[layer])

        self.longest_stimulus = inp.set_window(0, latest_time_end, padding=self.out_of_bound_strategy)  # hack: fake the longest stimulus
        
        return layer_activations
    
    def package_layer(self, activations, layer_spec, stimuli):
        layer_spec = "T" + layer_spec.replace('T', '')  # T has been moved to the first dimension
        return super().package_layer(activations, layer_spec, stimuli) 
    
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