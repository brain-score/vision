import numpy as np

from result_caching import store
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.model_interface import BrainModel
from brainscore.benchmarks.screen import place_on_screen
from brainio_collection.fetch import get_stimulus_set
# from brainscore.benchmarks.trials import repeat_trials, average_trials
from brainio_base.assemblies import DataAssembly

BLANK_STIM_NAME = 'dicarlo.Marques2020_blank'
RF_STIM_NAME = 'dicarlo.Marques2020_rf'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'

RF_NUMBER_OF_TRIALS = 10
ORIENTATION_NUMBER_OF_TRIALS = 40
TIMEBINS = {'V1': [(70, 170)]}
RF_THRSH = 0.05
RF_DELTA = 0.15
MEDIAN_MAX_RESP = {'V1': 33.8}
MEDIAN_SPONTANEOUS = {'V1': 0.82}


class PropertiesBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, neuronal_property, similarity_metric, **kwargs):
        super(PropertiesBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._neuronal_property = neuronal_property
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        self._number_of_trials = int(self._assembly.attrs['number_of_trials'])
        self._visual_degrees = self._assembly.stimulus_set['degrees']
        self.timebins = TIMEBINS[self.region]

    def __call__(self, model_identifier, model: BrainModel):
        model.start_recording(self.region, time_bins=self.timebins)
        stim_pos = get_stim_pos(self._assembly.stimulus_set)
        in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=self.region,
                                        pos=stim_pos)

        responses = get_firing_rates(model_identifier, model, self.region, self._assembly.stimulus_set.identifier,
                                     self._number_of_trials, in_rf)
        baseline = get_firing_rates(model_identifier, model, self.region, BLANK_STIM_NAME, self._number_of_trials,
                                    in_rf)

        model_property = self._neuronal_property(model_identifier=model_identifier, responses=responses,
                                                    baseline=baseline)
        raw_score = self._similarity_metric(model_property, self._assembly)
        ceiling = self._ceiling_func(self._assembly)
        return ceil_score(raw_score, ceiling)


@store(identifier_ignore=['model', 'in_rf'])
def get_firing_rates(model_identifier, model, region, stimulus_identifier, number_of_trials, in_rf):
    affine_transformation = firing_rates_affine(model_identifier=model_identifier, model=model, region=region)
    affine_transformation = affine_transformation.values

    activations = get_activations(model, stimulus_identifier, number_of_trials)
    activations = activations[in_rf]
    activations = affine_transformation[0] * activations + affine_transformation[1]
    activations.values[activations.values < 0] = 0
    return activations


def get_activations(model: BrainModel, stimulus_identifier, number_of_trials):
    stimulus_set = get_stimulus_set(stimulus_identifier)
    stimulus_set = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees())
    # stimulus_set = repeat_trials(stimulus_set, number_of_trials=number_of_trials) # coment
    activations = model.look_at(stimulus_set, number_of_trials) #, number_of_trials)
    if 'time_bin' in activations.dims:
        activations = activations.squeeze('time_bin')  # static case for these benchmarks
    # activations = average_trials(activations)  # coment
    return activations


def get_stim_pos(activations):
    pos_y = np.array(sorted(set(activations.pos_y.values)))
    pos_x = np.array(sorted(set(activations.pos_x.values)))
    assert len(pos_x) == 1 and len(pos_y) == 1
    return np.array([pos_y[0], pos_x[0]])


def filter_receptive_fields(model_identifier, model, region, pos, rf_delta=RF_DELTA):
    rf_pos, rf_map = map_receptive_field_locations(model_identifier=model_identifier, model=model, region=region)
    rf_pos = rf_pos.values
    d = np.linalg.norm(rf_pos - pos, axis=1)
    in_rf = np.squeeze(np.argwhere(d <= rf_delta))
    return in_rf


@store(identifier_ignore=['model'])
def map_receptive_field_locations(model_identifier, model: BrainModel, region):
    blank_activations = get_activations(model, BLANK_STIM_NAME, RF_NUMBER_OF_TRIALS)
    rf_activations = get_activations(model, RF_STIM_NAME, RF_NUMBER_OF_TRIALS)

    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    _assert_grating_activations(rf_activations)

    pos_y = np.array(sorted(set(rf_activations.pos_y.values)))
    pos_x = np.array(sorted(set(rf_activations.pos_x.values)))
    n_neuroids = rf_activations.values.shape[0]
    neuroid_ids = rf_activations.neuroid.values
    rf_activations = rf_activations.values
    rf_activations[rf_activations < 0] = 0

    rf_activations = rf_activations.reshape(n_neuroids, len(pos_y), len(pos_x), -1)
    rf_activations = rf_activations - np.reshape(blank_activations, [n_neuroids] +
                                                 [1] * (len(rf_activations.shape) - 1))

    rf_map = rf_activations.max(axis=3)

    rf_map[rf_map < 0] = 0

    max_resp = np.max(rf_map.reshape(n_neuroids, -1), axis=1)

    rf_pos = np.zeros((n_neuroids, 2))
    rf_pos[:] = np.nan

    for n in range(n_neuroids):
        exc_pos = rf_map[n] > max_resp[n] * RF_THRSH

        if max_resp[n] > 0:
            # rf centroid
            rf_coord = np.sum(
                np.argwhere(exc_pos) * np.repeat(np.expand_dims(rf_map[n, exc_pos], axis=1), 2, axis=1),
                axis=0) / np.sum(np.repeat(np.expand_dims(rf_map[n, exc_pos], axis=1), 2, axis=1), axis=0)
            # interpolates pos of rf centroid
            rf_pos[n, 0] = np.interp(rf_coord[0], np.arange(len(pos_y)), pos_y)
            rf_pos[n, 1] = np.interp(rf_coord[1], np.arange(len(pos_x)), pos_x)

    rf_pos = DataAssembly(rf_pos, coords={'neuroid': neuroid_ids, 'axis': ['y', 'x']}, dims=['neuroid', 'axis'])
    rf_map = DataAssembly(rf_map, coords={'neuroid': neuroid_ids, 'pos_y': pos_y, 'pos_x': pos_x},
                          dims=['neuroid', 'pos_y', 'pos_x'])

    return rf_pos, rf_map


@store(identifier_ignore=['model'])
def firing_rates_affine(model_identifier, model: BrainModel, region):
    blank_activations = get_activations(model, BLANK_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)
    orientation_activations = get_activations(model, ORIENTATION_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)

    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    _assert_grating_activations(orientation_activations)

    stim_pos = get_stim_pos(orientation_activations)

    rf_pos, rf_map = map_receptive_field_locations(model_identifier=model_identifier, model=model, region=region)
    rf_pos = rf_pos.values
    in_rf = filter_receptive_fields(rf_pos=rf_pos, pos=stim_pos, rf_delta=RF_DELTA)
    n_neuroids = len(in_rf)

    radius = sorted(set(orientation_activations.radius.values))
    sf = sorted(set(orientation_activations.sf.values))
    orientation = sorted(set(orientation_activations.orientation.values))
    phase = sorted(set(orientation_activations.phase.values))

    orientation_activations = orientation_activations.values
    orientation_activations[orientation_activations < 0] = 0

    blank_activations = blank_activations[in_rf]
    orientation_activations = orientation_activations[in_rf]
    orientation_activations = orientation_activations.reshape((n_neuroids, len(radius), len(sf),
                                                               len(orientation), len(phase)))
    orientation_activations = orientation_activations.mean(axis=4)
    orientation_activations = np.concatenate(
        (orientation_activations[:, 0, 2, :], orientation_activations[:, 1, 1, :],
         orientation_activations[:, 2, 0, :]), axis=1)
    orientation_activations = orientation_activations.max(axis=1)

    median_spontaneous = np.median(blank_activations)
    median_max_response = np.median(orientation_activations)

    slope = (MEDIAN_MAX_RESP[region] - MEDIAN_SPONTANEOUS[region]) / (median_max_response - median_spontaneous)
    offset = MEDIAN_SPONTANEOUS[region] - slope * median_spontaneous

    affine_transformation = np.array([slope, offset])
    affine_transformation = DataAssembly(affine_transformation)

    return affine_transformation


def _assert_grating_activations(activations):
    pos_y = np.array(sorted(set(activations.pos_y.values)))
    pos_x = np.array(sorted(set(activations.pos_x.values)))
    contrast = np.array(sorted(set(activations.contrast.values)))
    radius = np.array(sorted(set(activations.radius.values)))
    sf = np.array(sorted(set(activations.sf.values)))
    orientation = np.array(sorted(set(activations.orientation.values)))
    phase = np.array(sorted(set(activations.phase.values)))
    nStim = activations.values.shape[1]

    assert np.sum(np.tile(phase, len(pos_y) * len(pos_x) * len(contrast) * len(radius) * len(sf) * len(orientation)) ==
                  activations.phase.values) == nStim
    assert np.sum(np.tile(np.repeat(orientation, len(phase)), len(pos_y) * len(pos_x) * len(contrast) * len(radius) *
                          len(sf)) == activations.orientation.values) == nStim
    assert np.sum(np.tile(np.repeat(sf, len(phase) * len(orientation)), len(pos_y) * len(pos_x) * len(contrast) *
                          len(radius)) == activations.sf.values) == nStim
    assert np.sum(np.tile(np.repeat(radius, len(phase) * len(orientation) * len(sf)), len(pos_y) * len(pos_x) *
                          len(contrast)) == activations.radius.values) == nStim
    assert np.sum(np.tile(np.repeat(contrast, len(phase) * len(orientation) * len(sf) * len(radius)), len(pos_y) *
                          len(pos_x)) == activations.contrast.values) == nStim
    assert np.sum(np.tile(np.repeat(pos_x, len(phase) * len(orientation) * len(sf) * len(radius) * len(contrast)),
                          len(pos_y)) == activations.pos_x.values) == nStim
    assert np.sum(np.repeat(pos_y, len(phase) * len(orientation) * len(sf) * len(radius) * len(contrast) * len(pos_x))
                  == activations.pos_y.values) == nStim
