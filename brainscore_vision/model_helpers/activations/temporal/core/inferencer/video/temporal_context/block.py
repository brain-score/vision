import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from .base import TemporalContextInferencerBase
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding


class BlockInferencer(TemporalContextInferencerBase):
    """Inferencer that divides the original video into smaller blocks and does inference on the blocks separately.
    Finally, the activations are joint along the temporal dimension for the final activations. 

    Specifically, suppose the video lasts for 1000ms and the block size is 400ms.
    Then, the video is segmented into [0~400ms], [400~800ms], [800~1200ms] (1000~1200ms padded).
    The activations for each segment will be stacked together.

    The block size is determined by the temporal parameters (num_frames & duration) and temporal_context_strategy.
    If num_frames or duration is given, the model's temporal context will be set to match the two.
    """
    
    def inference(self, stimuli, layers):
        _, context = self._compute_temporal_context()

        if np.isinf(context):
            return super().inference(stimuli, layers)
        
        num_clips = []
        latest_time_end = 0
        for inp in stimuli:
            duration = inp.duration
            videos = []
            for time_start in np.arange(0, duration, context):
                time_end = time_start + context
                clip = inp.set_window(time_start, time_end, padding=self.out_of_bound_strategy)
                latest_time_end = max(latest_time_end, time_end)
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
                    video_activations = [np.moveaxis(a, time_index, 0) for a in video_activations]
                    layer_activations.setdefault(layer, []).append(np.concatenate(video_activations, axis=0))
                else:
                    layer_activations.setdefault(layer, []).append(np.stack(video_activations, axis=0))
                clip_start += num_clip

        for layer in layers:
            layer_activations[layer] = stack_with_nan_padding(layer_activations[layer])

        self.longest_stimulus = inp.set_window(0, latest_time_end, padding=self.out_of_bound_strategy)  # hack: fake the longest stimulus

        return layer_activations
    
    def package_layer(self, activations, layer, layer_spec, stimuli):
        layer_spec = "T" + layer_spec.replace('T', '')  # T has been moved to the first dimension
        return super().package_layer(activations, layer, layer_spec, stimuli) 