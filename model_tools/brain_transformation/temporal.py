import logging

import numpy as np

from typing import List, Tuple

from brainio_base.assemblies import merge_data_arrays

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
        self.commit = self._layer_model.commit
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
        responses = self._layer_model.look_at(stimuli)
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
        return responses

    @property
    def identifier(self):
        return self._layer_model.identifier
