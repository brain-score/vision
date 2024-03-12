import numpy as np
from collections import OrderedDict

from .base import TemporalInferencer
from ...executor import stack_with_nan_padding
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video


class CausalInferencer(TemporalInferencer):
    """Inferencer that ensures the activations are causal by feeding in the video in a causal manner. 

    Specifically, suppose the video lasts for 1000ms and the model samples every 100ms.
    Then, the activations from the last time step of the activations from the following videos:
    [0 ~ 100ms], [0 ~ 200ms], ..., [0 ~ 1000ms].
    will be stacked together to form the final activations for the video. In this way, the 
    activations are always "causal", which means that the temporal contexts of the model are 
    always from the past, not the future.

    If num_frames or duration is given, the model's temporal context will be set to match the two.
    Set
    
    Parameters
    ----------
    temporal_context_strategy: str
        specify how the length of temporal context for causal inference is determined.
        Options:
        - "greedy": the length of the temporal context is determined by the maximum of num_frames and duration.
        - "conservative": the length of the temporal context is determined by the minimum of num_frames and duration.
        - "fix": the length of the temporal context is determined by the specified "fixed_temporal_context".
    
    fixed_temporal_context: float
        specify the fixed length of the temporal context, in ms. It will be used only if temporal_context_strategy is "fix".
    
    out_of_bound_strategy: str
        specify how to handle the out-of-bound temporal context.
        Options:
        - "repeat": the out-of-bound temporal context will be repeated.
        - TODO: "black": the out-of-bound temporal context will be zero-padded.
    """
    def __init__(
            self, 
            *args,
            temporal_context_strategy : str = "greedy",
            fixed_temporal_context : float = None,
            out_of_bound_strategy : str = "repeat",
            **kwargs
        ):
        self.temporal_context_strategy = temporal_context_strategy
        self.fixed_temporal_context = fixed_temporal_context
        self.out_of_bound_strategy = out_of_bound_strategy
        if self.temporal_context_strategy == "fix" and self.fixed_temporal_context is None:
            raise ValueError("fixed_temporal_context must be specified if temporal_context_strategy is 'fix'.")
        if "time_alignment" in kwargs and kwargs["time_alignment"] != "per_frame_aligned":
            raise ValueError("CausalInferencer enforces time_alignment='per_frame_aligned'.")
        super().__init__(*args, **kwargs, time_alignment="per_frame_aligned")

    @property
    def identifier(self):
        to_add = f".strategy={self.temporal_context_strategy}.context={self._compute_temporal_context()}"
        return f"{super().identifier}{to_add}"
    
    def convert_paths(self, paths):
        videos = []
        for path in paths:
            if self.convert_to_video:
                video = Video.from_img(path, self.img_duration, self.fps)
            else:
                video = Video.from_path(path)
            videos.append(video)
        videos = [video.set_fps(self.fps) for video in videos]
        # do not check here.
        return videos

    def _get_implied_context(self, duration_or_num_frames, strategy):
        if duration_or_num_frames is None:
            return None

        if strategy == "greedy":
            return duration_or_num_frames[1]
        elif strategy == "conservative":
            return duration_or_num_frames[0]
        else:
            raise ValueError(f"Unknown temporal context strategy: {strategy}")
        
    def _overlapped_range(self, s1, e1, s2, e2):
        lower, upper = max(s1, s2), min(e1, e2)
        if lower > upper:
            raise ValueError(f"Ranges [{s1}, {e1}] and [{s2}, {e2}] do not overlap.")
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
            context = self.fixed_temporal_context
            assert ran[0] <= context <= ran[1], f"Fixed temporal context {context} is not within the range {ran}"

        else:
            raise ValueError(f"Unknown temporal context strategy: {strategy}")

        return lower, context
    
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

    def inference(self, stimuli, layers):
        interval = 1000 / self.fps
        num_clips = []
        for inp in stimuli:
            duration = inp.duration
            videos = []
            # here we ensure that the covered time range at least include the whole duration
            for time_end in np.arange(interval, duration+interval, interval):
                # see if the model only receive limited context
                lower, context = self._compute_temporal_context()
                time_start = self._get_time_start(time_end, context, lower)
                videos.append(inp.set_window(time_start, time_end, padding=self.out_of_bound_strategy))

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

        return layer_activations
    
    def package_layer(self, activations, layer, layer_spec, stimuli):
        layer_spec = "T" + layer_spec.replace('T', '')  # T has been moved to the first dimension
        return super().package_layer(activations, layer, layer_spec, stimuli) 