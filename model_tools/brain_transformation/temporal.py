from typing import List, Tuple

import logging
import numpy as np

from brainio_base.assemblies import merge_data_arrays, walk_coords
from brainscore.model_interface import BrainModel
from model_tools.utils import fullname


class TemporalIgnore(BrainModel):
    """
    Always output the same prediction, regardless of time-bins.
    Duplicates the LayerMappedModel prediction across time.
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
