import numpy as np
import math
from collections import OrderedDict
from tqdm import tqdm

from .base import TemporalContextInferencerBase
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding, data_assembly_mmap
from brainio.assemblies import NeuroidAssembly


class BlockInferencer(TemporalContextInferencerBase):
    """Inferencer that divides the original video into smaller blocks and does inference on the blocks separately.
    Finally, the activations are joint along the temporal dimension for the final activations. 

    Specifically, suppose the video lasts for 1000ms and the block size is 400ms.
    Then, the video is segmented into [0~400ms], [400~800ms], [800~1200ms] (1000~1200ms padded).
    The activations for each segment will be stacked together.

    The block size is determined by the temporal parameters (num_frames & duration) and temporal_context_strategy.
    If num_frames or duration is given, the model's temporal context will be set to match the two.
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
            for block_id, time_start in enumerate(np.arange(0, duration+EPS, context)):
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
                        num_feats, neuroid_coords = self._get_neuroid_coords(layer_activation, self._remove_T(self.layer_activation_format))
                        data = data_assembly_mmap(mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype, fill_value=np.nan)
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