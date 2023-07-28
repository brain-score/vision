import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, Tuple

from brainio.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from brainscore.submission.utils import UniqueKeyDict
from brainscore.utils import LazyLoad
from candidate_models.base_models import cornet
from model_tools.brain_transformation import ModelCommitment, LayerMappedModel
from model_tools.brain_transformation.temporal import fix_timebin_naming

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

    def create_neural_layer_model(self, identifier, activations_model, layers, region_layer_map):
        return CORnetLayerModel(identifier=identifier,
                                activations_model=activations_model,
                                layers=layers,
                                region_layer_map=region_layer_map,
                                time_mapping=self.time_mapping)

    def __init__(self, *args, time_mapping: Dict[str, Dict[int, Tuple[int, int]]], layers, **kwargs):
        """
        :param time_mapping: mapping from region -> {model_timestep -> (time_bin_start, time_bin_end)}
        """
        self.time_mapping = time_mapping
        super(CORnetCommitment, self).__init__(*args, layers=layers, **kwargs)
        # deal with activations_model returning a time_bin
        for key, executor in self.behavior_model.mapping.items():
            executor.activations_model = TemporalIgnore(executor.activations_model)


class CORnetLayerModel(LayerMappedModel):
    def __init__(self, *args, time_mapping: Dict[str, Dict[int, Tuple[int, int]]], layers, **kwargs):
        """
        :param time_mapping: mapping from region -> {model_timestep -> (time_bin_start, time_bin_end)}
        """
        kwargs.setdefault('region_layer_map', {target: [layer for layer in layers if layer.startswith(target)]
                                               for target in ['V1', 'V2', 'V4', 'IT']})
        super(CORnetLayerModel, self).__init__(*args, **kwargs)
        self.layers = layers
        self.recording_target = None
        self.time_mapping = time_mapping
        self.recording_time_bins = None

    def start_recording(self, recording_target, time_bins):
        self.recording_target = recording_target
        self.recording_time_bins = time_bins

    def look_at(self, stimuli, number_of_trials=1):  # ignore number_of_trials
        recording_layer = self.region_layer_map[self.recording_target]
        recording_layers = [layer for layer in self.layers if layer.startswith(recording_layer)]
        responses = self.activations_model(stimuli, layers=recording_layers)
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


def cornet_z_brainmodel(*args, **kwargs):
    kwargs.setdefault('identifier', 'CORnet-Z')
    return CORnetCommitment(*args,
                            activations_model=cornet('CORnet-Z'),
                            layers=[f'{region}.output-t0' for region in ['V1', 'V2', 'V4', 'IT']] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 150)},
                                'V2': {0: (70, 170)},
                                'V4': {0: (90, 190)},
                                'IT': {0: (100, 200)},
                            },
                            **kwargs)


def cornet_s_brainmodel(*args, **kwargs):
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = CORNET_S_TIMEMAPPING
    kwargs.setdefault('identifier', 'CORnet-S')
    return CORnetCommitment(*args,
                            activations_model=cornet('CORnet-S'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping=_build_time_mappings(time_mappings),
                            **kwargs)


def cornet_s222_brainmodel():
    time_start, time_step_size = 70, 100
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 2)}
    return CORnetCommitment(identifier='CORnet-S222', activations_model=cornet('CORnet-S222'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s101010_brainmodel():
    time_step_size = 20
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 10)}
    return CORnetCommitment(identifier='CORnet-S10', activations_model=cornet('CORnet-S10'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(10)), ('V4', range(10)), ('IT', range(10))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s444_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S444', activations_model=cornet('CORnet-S444'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(4)), ('IT', range(4))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s484_brainmodel():
    time_step_size = 50
    time_start = 70
    time_mapping = {timestep: (time_start + timestep * time_step_size, time_start + (timestep + 1) * time_step_size)
                    for timestep in range(0, 4)}
    return CORnetCommitment(identifier='CORnet-S484', activations_model=cornet('CORnet-S484'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(4)), ('V4', range(8)), ('IT', range(4))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_s10rep_brainmodel():
    activations_model = cornet('CORnet-S')
    old_times = activations_model._model.IT.times
    new_times = 10
    activations_model._model.IT.times = new_times
    size_12 = activations_model._model.IT.norm1_0.num_features
    size_3 = activations_model._model.IT.norm3_0.num_features
    for t in range(old_times, new_times):
        setattr(activations_model._model.IT, f'norm1_{t}', nn.BatchNorm2d(size_12))
        setattr(activations_model._model.IT, f'norm2_{t}', nn.BatchNorm2d(size_12))
        setattr(activations_model._model.IT, f'norm3_{t}', nn.BatchNorm2d(size_3))
    identifier = f'CORnet-S{new_times}rep'
    activations_model.identifier = identifier
    time_step_size = 10
    time_mapping = {timestep: (70 + timestep * time_step_size, 70 + (timestep + 1) * time_step_size)
                    for timestep in range(0, new_times)}
    return CORnetCommitment(identifier=identifier, activations_model=activations_model,
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r_brainmodel():
    return CORnetCommitment(identifier='CORnet-R', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {1: (60, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {1: (70, 110), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {1: (70, 110), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                            })


def cornet_r_ITt0_brainmodel():
    return CORnetCommitment(identifier='CORnet-R_ITt0', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (70, 100), 1: (100, 130), 2: (130, 160), 3: (160, 190), 4: (190, 250)},
                            })


def cornet_r_ITt1_brainmodel():
    return CORnetCommitment(identifier='CORnet-R_ITt1', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (10, 70), 1: (70, 170), 2: (170, 190), 3: (190, 210), 4: (210, 250)},
                            })


def cornet_r_ITt2_brainmodel():
    return CORnetCommitment(identifier='CORnet-R_ITt2', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (10, 30), 1: (30, 70), 2: (70, 170), 3: (170, 200), 4: (200, 250)},
                            })


def cornet_r_ITt3_brainmodel():
    return CORnetCommitment(identifier='CORnet-R_ITt3', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (10, 30), 1: (30, 50), 2: (50, 70), 3: (70, 150), 4: (150, 250)},
                            })


def cornet_r_ITt4_brainmodel():
    return CORnetCommitment(identifier='CORnet-R_ITt4', activations_model=cornet('CORnet-R'),
                            layers=[f'{area}.output-t{timestep}' for area in ['V1', 'V2', 'V4', 'IT'] for timestep in
                                    range(5)] + ['decoder.avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (10, 30), 1: (30, 50), 2: (50, 60), 3: (60, 70), 4: (70, 250)},
                            })


def cornet_r10rep_brainmodel():
    activations_model = cornet('CORnet-R')
    new_times = 10
    activations_model._model.times = new_times
    activations_model.identifier = f'CORnet-R{new_times}'
    time_step_size = 10
    time_mapping = {timestep: (70 + timestep * time_step_size, 70 + (timestep + 1) * time_step_size)
                    for timestep in range(0, new_times)}
    return CORnetCommitment(identifier=f'CORnet-R{new_times}', activations_model=activations_model,
                            layers=['maxpool-t0'] +
                                   [f'{area}.relu3-t{timestep}' for area in ['block2', 'block3', 'block4']
                                    for timestep in range(new_times)] + ['avgpool-t0'],
                            time_mapping={'IT': time_mapping})


def cornet_r2_brainmodel():
    return CORnetCommitment(identifier='CORnet-R2', activations_model=cornet('CORnet-R2'),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}' for area in ['V2', 'V4', 'IT']
                                    for timestep in range(5)] + ['avgpool-t0'],
                            time_mapping={
                                'V1': {0: (50, 150)},
                                'V2': {0: (50, 80), 1: (80, 110), 2: (110, 140), 3: (140, 170), 4: (170, 200)},
                                'V4': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                                'IT': {0: (70, 110), 1: (110, 140), 2: (140, 170), 3: (170, 200), 4: (200, 250)},
                            })


class CORnetBrainPool(UniqueKeyDict):
    def __init__(self):
        super(CORnetBrainPool, self).__init__(reload=True)

        model_pool = {
            'CORnet-Z': LazyLoad(cornet_z_brainmodel),
            'CORnet-S': LazyLoad(cornet_s_brainmodel),
            'CORnet-S101010': LazyLoad(cornet_s101010_brainmodel),
            'CORnet-S222': LazyLoad(cornet_s222_brainmodel),
            'CORnet-S444': LazyLoad(cornet_s444_brainmodel),
            'CORnet-S484': LazyLoad(cornet_s484_brainmodel),
            'CORnet-S10rep': LazyLoad(cornet_s10rep_brainmodel),
            'CORnet-R': LazyLoad(cornet_r_brainmodel),
            'CORnet-R_ITt0': LazyLoad(cornet_r_ITt0_brainmodel),
            'CORnet-R_ITt1': LazyLoad(cornet_r_ITt1_brainmodel),
            'CORnet-R_ITt2': LazyLoad(cornet_r_ITt2_brainmodel),
            'CORnet-R_ITt3': LazyLoad(cornet_r_ITt3_brainmodel),
            'CORnet-R_ITt4': LazyLoad(cornet_r_ITt4_brainmodel),
            'CORnet-R10rep': LazyLoad(cornet_r10rep_brainmodel),
            'CORnet-R2': LazyLoad(cornet_r2_brainmodel),
        }

        self._accessed_brain_models = []

        for identifier, brain_model in model_pool.items():
            self[identifier] = brain_model


cornet_brain_pool = CORnetBrainPool()
