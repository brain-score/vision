import logging
import numpy as np
from typing import List, Tuple

from brainio.assemblies import merge_data_arrays, walk_coords
from brainscore_vision.model_helpers.utils import fullname
from brainscore_vision.model_interface import BrainModel


def iterable_to_list(arr):
    """ recursively converts a list, tuple, or numpy array into a python list. """
    if isinstance(arr, (list, tuple, np.ndarray)):
        return [iterable_to_list(a) for a in arr]
    else:
        return arr


def time_align(source_time_bins: List[Tuple[int, int]], target_time_bins: List[Tuple[int, int]], mode: str = "portion"):
    """ return the aligned binary indicator in the source.
        belong_to matrix: (target_time_bin, source_time_bin)
          1 if the target time bin covers the source time bin
          0 otherwise
          can be a portion if mode=="portion"

        NOTE: here we assume the source time bins are contiguous, i.e. no gap between them.

        Example:
        source_time_bins = [(0, 100), (100, 200), (200, 300)]
        target_time_bins = [(0, 50), (250, 300)]

        mode = "center"
        belong_to = [[1, 0, 0], [0, 0, 1]]

        mode = "portion"
        belong_to = [[0.5, 0, 0], [0, 0, 0.5]]
    """
    
    source_time_bins = np.array(iterable_to_list(source_time_bins))  # otherwise object array [(a,b), (c,d)...]
    target_time_bins = np.array(iterable_to_list(target_time_bins))    
    assert (source_time_bins[:, 0] <= source_time_bins[:, 1]).all()
    assert (target_time_bins[:, 0] <= target_time_bins[:, 1]).all()

    target_time_starts = target_time_bins[:, 0]
    target_time_ends = target_time_bins[:, 1]

    source_time_starts = source_time_bins[:, 0]
    source_time_ends = source_time_bins[:, 1]

    if mode == "center":

        target_time_mids = target_time_bins.mean(-1)
        belong_to = (target_time_mids.reshape(-1, 1) >= source_time_starts) & (target_time_mids.reshape(-1, 1) < source_time_ends)
        belong_to = belong_to.astype(int)

    elif mode == "portion":

        target_time_starts = target_time_starts.reshape(-1, 1)
        target_time_ends = target_time_ends.reshape(-1, 1)
        source_time_starts = source_time_starts.reshape(1, -1)
        source_time_ends = source_time_ends.reshape(1, -1)

        # overlap
        overlap_starts = np.maximum(target_time_starts, source_time_starts)
        overlap_ends = np.minimum(target_time_ends, source_time_ends)
        overlap = np.maximum(overlap_ends - overlap_starts, 0)

        # target time bin size
        source_time_size = source_time_ends - source_time_starts

        # portion
        belong_to = overlap / source_time_size

        # get target time bin whose range is not covered completely
        not_completed_covered = (overlap_ends.max(1) < target_time_ends.T) | (overlap_starts.min(1) > target_time_starts.T)
        not_completed_covered = not_completed_covered[0]
        belong_to[not_completed_covered] = 0

    else:
        raise NotImplementedError("Temporal alignment mode should be either 'center' or 'portion'.")

    return belong_to


def assembly_time_align(source, target_time_bins, mode="portion"):
    assert hasattr(source, "time_bin")
    assert source.time_bin.variable.level_names == ['time_bin_start', 'time_bin_end']
    assert len(target_time_bins[0]) == 2
    source_time_bins = np.array(iterable_to_list(source.time_bin.values))  # otherwise object array [(a,b), (c,d)...]
    target_time_bins = np.array(iterable_to_list(target_time_bins)) 

    belong_to = time_align(source_time_bins, target_time_bins, mode)
    invalid = np.where(belong_to.sum(1)==0)[0]
    assert len(invalid)==0, f"Target time bin(s):\n{target_time_bins[invalid]} invalid. The source time bins are:\n{source_time_bins}."
    source = source.transpose(..., "time_bin")
    source_data = source.values
    ret_data = []
    for source_belong_to in belong_to:
        relevant = source_belong_to>0
        data = source_data[..., relevant]
        weights = source_belong_to[relevant]
        data = (data * weights).sum(-1) / weights.sum()  # weighted average
        ret_data.append(data)
    ret_data = np.stack(ret_data, -1)  # can be memory-intensive

    # create assembly
    coords = {k: v for k, v in source.coords.items() if k != "time_bin"}
    ret = type(source)(
        ret_data,
        dims=source.dims,
        coords={
            "time_bin_start": ("time_bin", target_time_bins[:, 0]),
            "time_bin_end": ("time_bin", target_time_bins[:, 1]),
            **coords
        },
    )
    return ret


class TemporalAligned(BrainModel):
    """
    Deals with the alignment of time-bins.
    
    If the underlying model does not provide a time dimension in its predictions, this always outputs the same prediction for all requested recording time-bins. More specifically, this duplicates the LayerMappedModel prediction across time.
    
    If the underlying model does provide a time dimension in its predictions, align those time points to the requested recording time-bins.
    """

    def __init__(self, layer_model):
        self._logger = logging.getLogger(fullname(self))
        self._layer_model = layer_model
        self.region_layer_map = self._layer_model.region_layer_map
        self.activations_model = self._layer_model.activations_model
        self.start_task = self._layer_model.start_task
        self._time_bins = None

    def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
        self._layer_model.start_recording(recording_target)
        self._time_bins = time_bins

    def visual_degrees(self) -> int:
        return self._layer_model.visual_degrees()

    def look_at(self, stimuli, number_of_trials=1):
        responses = self._layer_model.look_at(stimuli, number_of_trials=number_of_trials)
        if "time_bin" not in responses.dims:
            # if the model does not support temporal output, assume the output is the same across time
            time_responses = []
            self._logger.debug(f'Repeating single assembly across time bins {self._time_bins}')
            for time_bin in self._time_bins:
                time_bin = time_bin if not isinstance(time_bin, np.ndarray) else time_bin.tolist()
                time_bin_start, time_bin_end = time_bin
                bin_responses = responses.expand_dims('time_bin_start').expand_dims('time_bin_end')
                bin_responses['time_bin_start'] = [time_bin_start]
                bin_responses['time_bin_end'] = [time_bin_end]
                bin_responses = bin_responses.stack(time_bin=['time_bin_start', 'time_bin_end'])
                time_responses.append(bin_responses)
            responses = merge_data_arrays(time_responses)
        else:
            # for temporal models, align the time bins
            responses = assembly_time_align(responses, self._time_bins)
            
        if len(self._time_bins) == 1:
            responses = responses.squeeze('time_bin')
        responses = fix_timebin_naming(responses)
        return responses

    @property
    def identifier(self):
        return self._layer_model.identifier


def fix_timebin_naming(assembly):
    """
    renames coordinate time_bin_level_0 to time_bin_start and time_bin_level_1 to time_bin_end
    to work around bug introduced in xarray 0.16.2 (and still present in 0.17.0).
    """
    # jjpr had found that xarray 0.16.2 introduced a bug where xarray.core.alignment._get_joiner assumes Index when the
    # object is a MultiIndex.
    # xarray.rename for some reason does not work for some reason, it cannot find the coords
    rename = dict(time_bin_level_0='time_bin_start', time_bin_level_1='time_bin_end')
    assembly = type(assembly)(assembly.values, coords={
        rename[coord] if coord in rename else coord: (dims, values)
        for coord, dims, values in walk_coords(assembly)}, dims=assembly.dims)
    return assembly
