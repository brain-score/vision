import numpy as np
from collections import OrderedDict

from .base import TemporalInferencer
from ..executor import stack_with_nan_padding


class CausalInferencer(TemporalInferencer):
    """Temporal inferencer that feeds to the model frame by frame.

    Specifically, suppose the video lasts for 1000ms and the model samples every 100ms.
    Then, the activations from the last time step of the activations from the following videos:
    [0 ~ 100ms], [0 ~ 200ms], ..., [0 ~ 1000ms].
    will be stacked together to form the final activations for the video. In this way, the 
    activations are always "causal", which means that the temporal contexts of the model are 
    always from the past, not the future.
    
    Parameters
    ----------
    max_temporal_context: int
        specify the maximum temporal context the model can take, in ms.
        If None, the model will take the whole duration from the beginning.

    """
    def __init__(self, *args, max_temporal_context=2000, **kwargs):
        self.max_temporal_context = max_temporal_context
        super().__init__(*args, **kwargs)

    def _compute_temporal_context(self):
        duration = self.duration
        num_frames = self.num_frames
        if self.max_temporal_context:
            if not duration is not None or num_frames is not None:
                duration = self.max_temporal_context
        
        if duration is not None or num_frames is not None:
            if num_frames is not None:  # prioritize num_frames
                context = num_frames * 1000 / self.fps
            else:
                context = duration
        else:
            context = None
        return context

    def inference(self, stimuli, layers):
        interval = 1000 / self.fps
        num_clips = []
        for inp in stimuli:
            duration = inp.duration
            videos = []
            for time_end in np.arange(duration, 0, -interval)[::-1]:
                # see if the model only receive limited context
                context = self._compute_temporal_context()
                time_start = time_end - context if context else 0
                videos.append(inp.set_window(time_start, time_end))

            self._executor.add_stimuli(videos)
            num_clips.append(len(videos))

        activations = self._executor.execute(layers)
        layer_activations = OrderedDict()
        for layer in layers:
            activation_dims = self.activations_spec[layer]
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