import logging
import numpy as np
from brainio.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.brain_transformation.temporal import fix_timebin_naming
from tqdm import tqdm
from typing import Dict, Tuple
from .cornet_helpers import cornet

_logger = logging.getLogger(__name__)

CORNET_S_TIMEMAPPING = {
    'V1': (50, 100, 1),
    'V2': (70, 100, 2),
    'V4': (90, 50, 4),
    'IT': (100, 100, 2),
}


class CORnetCommitment(ModelCommitment):
    """
    CORnet commitment where only the model interface is implemented and behavioral readouts are attached.
    Importantly, layer-region commitments do not occur due to the anatomical pre-mapping.
    Further, due to the temporal component of the model, requested time-bins are matched to the nearest committed
    time-bin for the model.
    """

    def __init__(self, *args, time_mapping: Dict[str, Dict[int, Tuple[int, int]]], **kwargs):
        """
        :param time_mapping: mapping from region -> {model_timestep -> (time_bin_start, time_bin_end)}
        """
        super(CORnetCommitment, self).__init__(*args, **kwargs)
        self.time_mapping = time_mapping
        self.recording_layers = None
        self.recording_time_bins = None
        # deal with activations_model returning a time_bin
        for key, executor in self.behavior_model.mapping.items():
            executor.activations_model = TemporalIgnore(executor.activations_model)

    def start_recording(self, recording_target, time_bins):
        self.recording_target = recording_target
        self.recording_layers = [layer for layer in self.layers if layer.startswith(recording_target)]
        self.recording_time_bins = time_bins

    def look_at(self, stimuli, number_of_trials=1):
        if self.do_behavior:
            return super(CORnetCommitment, self).look_at(stimuli, number_of_trials=number_of_trials)
        else:
            return self.look_at_temporal(stimuli=stimuli)  # ignore number_of_trials

    def look_at_temporal(self, stimuli):
        responses = self.activations_model(stimuli, layers=self.recording_layers)
        # map time
        if hasattr(self, 'recording_target'):
            regions = set([self.recording_target])
        else:
            regions = set(responses['region'].values)
        if len(regions) > 1:
            raise NotImplementedError("cannot handle more than one simultaneous region")
        region = list(regions)[0]
        time_bins = [self.time_mapping[region][timestep] if timestep in self.time_mapping[region] else (None, None)
                     for timestep in responses['time_step'].values]
        responses['time_bin_start'] = 'time_step', [time_bin[0] for time_bin in time_bins]
        responses['time_bin_end'] = 'time_step', [time_bin[1] for time_bin in time_bins]
        responses = NeuroidAssembly(responses.rename({'time_step': 'time_bin'}))
        responses = responses[{'time_bin': [not np.isnan(time_start) for time_start in responses['time_bin_start']]}]
        # select time
        time_responses = []
        for time_bin in tqdm(self.recording_time_bins, desc='CORnet-time to recording time'):
            time_bin = time_bin if not isinstance(time_bin, np.ndarray) else time_bin.tolist()
            time_bin_start, time_bin_end = time_bin
            nearest_start = find_nearest(responses['time_bin_start'].values, time_bin_start)
            bin_responses = responses.sel(time_bin_start=nearest_start)
            bin_responses = NeuroidAssembly(bin_responses.values, coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(bin_responses)
                   if coord not in ['time_bin_level_0', 'time_bin_end']},
                **{'time_bin_start': ('time_bin', [time_bin_start]),
                   'time_bin_end': ('time_bin', [time_bin_end])}
            }, dims=bin_responses.dims)
            time_responses.append(bin_responses)
        responses = merge_data_arrays(time_responses)
        responses = fix_timebin_naming(responses)  # work around xarray merge bug introduced in 0.16.2
        return responses


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class TemporalIgnore:
    """
    Wrapper around a activations model that squeezes out the temporal axis.
    Useful when there is only one time step and the behavioral readout does not know what to do with time.
    """

    def __init__(self, temporal_activations_model):
        self._activations_model = temporal_activations_model

    def __call__(self, *args, **kwargs):
        activations = self._activations_model(*args, **kwargs)
        activations = activations.squeeze('time_step')
        return activations


def _build_time_mappings(time_mappings):
    return {region: {
        timestep: (time_start + timestep * time_step_size,
                   time_start + (timestep + 1) * time_step_size)
        for timestep in range(0, timesteps)}
        for region, (time_start, time_step_size, timesteps) in time_mappings.items()}


def cornet_z_brainmodel():
    return CORnetCommitment(identifier='CORnet-Z', activations_model=cornet('CORnet-Z'),
                            layers=[f'{region}.output-t0' for region in ['V1', 'V2', 'V4', 'IT']] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 150)},
                                'V2': {0: (70, 170)},
                                'V4': {0: (90, 190)},
                                'IT': {0: (100, 200)},
                            })


def cornet_s_brainmodel():
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = CORNET_S_TIMEMAPPING
    return CORnetCommitment(identifier='CORnet-S', activations_model=cornet('CORnet-S'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping=_build_time_mappings(time_mappings))
