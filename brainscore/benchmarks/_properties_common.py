import numpy as np

from brainio_base.assemblies import DataAssembly
from brainscore import get_stimulus_set
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from result_caching import store

BLANK_STIM_NAME = 'dicarlo.Marques2020_blank'
RF_STIM_NAME = 'dicarlo.Marques2020_receptive_field'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'

RF_NUMBER_OF_TRIALS = 10
ORIENTATION_NUMBER_OF_TRIALS = 40
RF_THRSH = 0.05
RF_DELTA = 0.15
MEDIAN_MAX_RESP = {'V1': 33.8}
MEDIAN_SPONTANEOUS = {'V1': 0.82}


class PropertiesBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, neuronal_property, similarity_metric, timebins, **kwargs):
        super(PropertiesBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._neuronal_property = neuronal_property
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        self._number_of_trials = int(self._assembly.attrs['number_of_trials'])
        self._visual_degrees = self._assembly.stimulus_set['degrees']
        self.timebins = timebins

    def __call__(self, model: BrainModel):
        model_identifier = model.identifier
        model.start_recording(self.region, time_bins=self.timebins)
        stim_pos = get_stimulus_position(self._assembly.stimulus_set)
        in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=self.region,
                                        pos=stim_pos)

        responses = get_firing_rates(model_identifier=model_identifier, model=model, region=self.region,
                                     stimulus_identifier=self._assembly.stimulus_set.identifier,
                                     number_of_trials=self._number_of_trials, in_rf=in_rf)
        baseline = get_firing_rates(model_identifier=model_identifier, model=model, region=self.region,
                                    stimulus_identifier=BLANK_STIM_NAME,
                                    number_of_trials=self._number_of_trials, in_rf=in_rf)

        model_property = self._neuronal_property(model_identifier=model_identifier, responses=responses,
                                                 baseline=baseline)
        raw_score = self._similarity_metric(model_property, self._assembly)
        ceiling = self._ceiling_func(self._assembly)
        return ceil_score(raw_score, ceiling)


@store(identifier_ignore=['model', 'in_rf'])
def get_firing_rates(model_identifier, model, region, stimulus_identifier, number_of_trials, in_rf):
    affine_transformation = firing_rates_affine(model_identifier=model_identifier, model=model, region=region)
    affine_transformation = affine_transformation.values

    activations = record_from_model(model, stimulus_identifier, number_of_trials)
    activations = activations[in_rf]
    activations = affine_transformation[0] * activations + affine_transformation[1]
    activations.values[activations.values < 0] = 0
    return activations


def record_from_model(model: BrainModel, stimulus_identifier, number_of_trials):
    stimulus_set = get_stimulus_set(stimulus_identifier)
    stimulus_set = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees())
    activations = model.look_at(stimulus_set, number_of_trials)
    if 'time_bin' in activations.dims:
        activations = activations.squeeze('time_bin')  # static case for these benchmarks
    return activations


def get_stimulus_position(stimulus_set):
    position_y = np.array(sorted(set(stimulus_set.position_y.values)))
    position_x = np.array(sorted(set(stimulus_set.position_x.values)))
    assert len(position_x) == 1 and len(position_y) == 1
    return np.array([position_y[0], position_x[0]])


def filter_receptive_fields(model_identifier, model, region, pos, rf_delta=RF_DELTA):
    rf_pos, rf_map = map_receptive_field_locations(model_identifier=model_identifier, model=model, region=region)
    rf_pos = rf_pos.values
    d = np.linalg.norm(rf_pos - pos, axis=1)
    in_rf = np.squeeze(np.argwhere(d <= rf_delta))
    return in_rf


@store(identifier_ignore=['model'])
def map_receptive_field_locations(model_identifier, model: BrainModel, region):
    blank_activations = record_from_model(model, BLANK_STIM_NAME, RF_NUMBER_OF_TRIALS)
    rf_activations = record_from_model(model, RF_STIM_NAME, RF_NUMBER_OF_TRIALS)

    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    _assert_grating_activations(rf_activations)

    position_y = np.array(sorted(set(rf_activations.position_y.values)))
    position_x = np.array(sorted(set(rf_activations.position_x.values)))
    n_neuroids = rf_activations.values.shape[0]
    neuroid_ids = rf_activations.neuroid.values
    rf_activations = rf_activations.values
    rf_activations[rf_activations < 0] = 0

    rf_activations = rf_activations.reshape(n_neuroids, len(position_y), len(position_x), -1)
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
            rf_pos[n, 0] = np.interp(rf_coord[0], np.arange(len(position_y)), position_y)
            rf_pos[n, 1] = np.interp(rf_coord[1], np.arange(len(position_x)), position_x)

    rf_pos = DataAssembly(rf_pos, coords={'neuroid': neuroid_ids, 'axis': ['y', 'x']}, dims=['neuroid', 'axis'])
    rf_map = DataAssembly(rf_map, coords={'neuroid': neuroid_ids, 'position_y': position_y, 'position_x': position_x},
                          dims=['neuroid', 'position_y', 'position_x'])

    return rf_pos, rf_map


@store(identifier_ignore=['model'])
def firing_rates_affine(model_identifier, model: BrainModel, region):
    blank_activations = record_from_model(model, BLANK_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)
    orientation_activations = record_from_model(model, ORIENTATION_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)

    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    _assert_grating_activations(orientation_activations)

    stim_pos = get_stimulus_position(orientation_activations)

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
    position_y = np.array(sorted(set(activations.position_y.values)))
    position_x = np.array(sorted(set(activations.position_x.values)))
    contrast = np.array(sorted(set(activations.contrast.values)))
    radius = np.array(sorted(set(activations.radius.values)))
    spatial_frequency = np.array(sorted(set(activations.spatial_frequency.values)))
    orientation = np.array(sorted(set(activations.orientation.values)))
    phase = np.array(sorted(set(activations.phase.values)))
    nStim = activations.values.shape[1]

    assert np.sum(np.tile(phase, len(position_y) * len(position_x) * len(contrast) * len(radius) *
                          len(spatial_frequency) * len(orientation)) == activations.phase.values) == nStim
    assert np.sum(np.tile(np.repeat(orientation, len(phase)), len(position_y) * len(position_x) * len(contrast) *
                          len(radius) * len(spatial_frequency)) == activations.orientation.values) == nStim
    assert np.sum(np.tile(np.repeat(spatial_frequency, len(phase) * len(orientation)), len(position_y) *
                          len(position_x) * len(contrast) * len(radius)) == activations.sf.values) == nStim
    assert np.sum(np.tile(np.repeat(radius, len(phase) * len(orientation) * len(spatial_frequency)), len(position_y) *
                          len(position_x) * len(contrast)) == activations.radius.values) == nStim
    assert np.sum(np.tile(np.repeat(contrast, len(phase) * len(orientation) * len(spatial_frequency) * len(radius)),
                          len(position_y) * len(position_x)) == activations.contrast.values) == nStim
    assert np.sum(np.tile(np.repeat(position_x, len(phase) * len(orientation) * len(spatial_frequency) * len(radius) *
                                    len(contrast)), len(position_y)) == activations.position_x.values) == nStim
    assert np.sum(np.repeat(position_y, len(phase) * len(orientation) * len(spatial_frequency) * len(radius) *
                            len(contrast) * len(position_x)) == activations.position_y.values) == nStim


def calc_circular_variance(orientation_curve, orientation):
    vect_sum = orientation_curve.dot(np.exp(1j * 2 * orientation / 180 * np.pi))
    osi = np.absolute(vect_sum) / np.sum(np.absolute(orientation_curve))
    return 1 - osi


def calc_bandwidth(orientation_curve, orientation, filt_type='hanning', thrsh=0.5, mode='full'):
    from scipy.interpolate import UnivariateSpline
    or_ext = np.hstack((orientation - 180, orientation, orientation + 180))
    or_curve_ext = np.tile(orientation_curve, (1, 3))

    if filt_type == 'hanning':
        w = np.array([0, 2 / 5, 1, 2 / 5, 0])
    elif filt_type == 'flat':
        w = np.array([1, 1, 1, 1, 1])
    elif filt_type == 'smooth':
        w = np.array([0, 1 / 5, 1, 1 / 5, 0])

    if filt_type is not None:
        or_curve_ext = np.convolve(w / w.sum(), np.squeeze(or_curve_ext), mode='same')
    or_curve_spl = UnivariateSpline(or_ext, or_curve_ext, s=0.)

    or_full = np.linspace(-180, 359, 540)
    or_curve_full = or_curve_spl(or_full)
    pref_or_fit = np.argmax(or_curve_full[181:360])
    or_curve_max = or_curve_full[pref_or_fit + 181]

    try:
        less = np.where(or_curve_full <= or_curve_max * thrsh)[0][:]
        p1 = or_full[less[np.where(less < pref_or_fit + 181)[0][-1]]]
        p2 = or_full[less[np.where(less > pref_or_fit + 181)[0][0]]]
        bw = (p2 - p1)
        if bw > 180:
            bw = np.nan
    except:
        bw = np.nan
    if mode is 'half':
        bw = bw / 2
    return bw, pref_or_fit, or_full[181:360], or_curve_full[181:360]


def calc_orthogonal_preferred_ratio(orientation_curve, orientation):
    pref_orientation = np.argmax(orientation_curve)
    orth_orientation = pref_orientation + int(len(orientation) / 2)
    if orth_orientation >= len(orientation):
        orth_orientation -= len(orientation)
    opr = orientation_curve[orth_orientation] / orientation_curve[pref_orientation]
    return opr
