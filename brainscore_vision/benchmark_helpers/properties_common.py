import numpy as np

import brainscore_vision
from brainio.assemblies import DataAssembly
from brainscore_vision.benchmarks import BenchmarkBase, ceil_score
from .screen import place_on_screen
from brainscore_vision.model_interface import BrainModel
from result_caching import store

BLANK_STIM_NAME = 'Marques2020_blank'
RF_STIM_NAME = 'Marques2020_receptive_field'
ORIENTATION_STIM_NAME = 'Marques2020_orientation'

RF_NUMBER_OF_TRIALS = 10
ORIENTATION_NUMBER_OF_TRIALS = 20
RF_THRSH = 0.05
RF_DELTA = 0.15
MEDIAN_MAX_RESP = {'V1': 33.8}
MEDIAN_SPONTANEOUS = {'V1': 0.82}
SINGLE_MAX_RESP = {'V1': 243.1}
RESP_THRESH = {'V1': 5}
LOW_INTERVAL_MAX_RESP = {'V1': 11.14}
HIGH_INTERVAL_MAX_RESP = {'V1': 86.27}
LOW_INTERVAL_PERCENTILE = 10
HIGH_INTERVAL_PERCENTILE = 90


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

    @property
    def ceiling(self):
        return self._ceiling_func(self._assembly)



@store(identifier_ignore=['model', 'in_rf'])
def get_firing_rates(model_identifier, model, region, stimulus_identifier, number_of_trials, in_rf):
    affine_transformation = firing_rates_affine(model_identifier=model_identifier, model=model, region=region)
    affine_transformation = affine_transformation.values

    activations = record_from_model(model, stimulus_identifier, number_of_trials)
    activations = activations[in_rf]
    activations.values[activations.values < 0] = 0

    activations = affine_transformation[0] * activations + affine_transformation[1]
    activations.values[activations.values < 0] = 0
    return activations


def record_from_model(model: BrainModel, stimulus_identifier, number_of_trials):
    stimulus_set = brainscore_vision.load_stimulus_set(stimulus_identifier)
    stimulus_set = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees())
    activations = model.look_at(stimulus_set, number_of_trials)
    if 'time_bin' in activations.dims:
        activations = activations.squeeze('time_bin')  # static case for these benchmarks
    if not activations.values.flags['WRITEABLE']:
        activations.values.setflags(write=1)
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
    # model_identifier: used to differentiate between different models when caching results.

    blank_activations = record_from_model(model, BLANK_STIM_NAME, RF_NUMBER_OF_TRIALS)
    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    rf_activations = record_from_model(model, RF_STIM_NAME, RF_NUMBER_OF_TRIALS)

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
    # model_identifier: used to differentiate between different models when caching results.
    
    blank_activations = record_from_model(model, BLANK_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)
    orientation_activations = record_from_model(model, ORIENTATION_STIM_NAME, ORIENTATION_NUMBER_OF_TRIALS)

    blank_activations = blank_activations.values
    blank_activations[blank_activations < 0] = 0

    _assert_grating_activations(orientation_activations)

    stim_pos = get_stimulus_position(orientation_activations)

    in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region, pos=stim_pos)
    n_neuroids = len(in_rf)

    spatial_frequency = sorted(set(orientation_activations.spatial_frequency.values))
    orientation = sorted(set(orientation_activations.orientation.values))
    phase = sorted(set(orientation_activations.phase.values))
    nStim = orientation_activations.values.shape[1]
    n_cycles = nStim // (len(phase) * len(orientation) * len(spatial_frequency))

    orientation_activations = orientation_activations.values
    orientation_activations[orientation_activations < 0] = 0

    blank_activations = blank_activations[in_rf]
    orientation_activations = orientation_activations[in_rf]
    orientation_activations = orientation_activations.reshape((n_neuroids, n_cycles, len(spatial_frequency),
                                                               len(orientation), len(phase)))
    orientation_activations = orientation_activations.mean(axis=4).reshape((n_neuroids, -1)).max(axis=1)

    responsive_neurons = (orientation_activations - blank_activations[:, 0]) >  \
                         (RESP_THRESH[region] / SINGLE_MAX_RESP[region]) * \
                         np.max(orientation_activations - blank_activations[:, 0])

    median_baseline = np.median(blank_activations[responsive_neurons])
    median_activations = np.median(orientation_activations[responsive_neurons])

    slope = (MEDIAN_MAX_RESP[region] - MEDIAN_SPONTANEOUS[region]) / \
            (median_activations - median_baseline)
    offset = MEDIAN_SPONTANEOUS[region] - slope * median_baseline

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

    if nStim == len(position_x) * len(position_y) * len(contrast) * len(radius) * len(spatial_frequency) * \
            len(orientation) * len(phase):
        assert np.sum(np.tile(phase, len(position_y) * len(position_x) * len(contrast) * len(radius) *
                              len(spatial_frequency) * len(orientation)) == activations.phase.values) == nStim
        assert np.sum(np.tile(np.repeat(orientation, len(phase)), len(position_y) * len(position_x) * len(contrast) *
                              len(radius) * len(spatial_frequency)) == activations.orientation.values) == nStim
        assert np.sum(np.tile(np.repeat(spatial_frequency, len(phase) * len(orientation)), len(position_y) *
                              len(position_x) * len(contrast) * len(radius)) == activations.spatial_frequency.values) == nStim
        assert np.sum(np.tile(np.repeat(radius, len(phase) * len(orientation) * len(spatial_frequency)), len(position_y) *
                              len(position_x) * len(contrast)) == activations.radius.values) == nStim
        assert np.sum(np.tile(np.repeat(contrast, len(phase) * len(orientation) * len(spatial_frequency) * len(radius)),
                              len(position_y) * len(position_x)) == activations.contrast.values) == nStim
        assert np.sum(np.tile(np.repeat(position_x, len(phase) * len(orientation) * len(spatial_frequency) * len(radius) *
                                        len(contrast)), len(position_y)) == activations.position_x.values) == nStim
        assert np.sum(np.repeat(position_y, len(phase) * len(orientation) * len(spatial_frequency) * len(radius) *
                                len(contrast) * len(position_x)) == activations.position_y.values) == nStim
    else:
        n_cycles = nStim // (len(phase) * len(orientation) * len(spatial_frequency))
        assert np.sum(np.tile(phase, n_cycles * len(spatial_frequency) * len(orientation)) == activations.phase.values)\
               == nStim
        assert np.sum(np.tile(np.repeat(orientation, len(phase)), n_cycles * len(spatial_frequency)) ==
                      activations.orientation.values) == nStim
        assert np.sum(np.tile(np.repeat(spatial_frequency, len(phase) * len(orientation)), n_cycles) ==
                      activations.spatial_frequency.values) == nStim


def _assert_texture_activations(activations):
    activations = activations.sortby(['type', 'family', 'sample'])

    type = np.array(sorted(set(activations.type.values)))
    family = np.array(sorted(set(activations.family.values)))
    sample = np.array(sorted(set(activations.sample.values)))

    n_type = len(type)
    n_family = len(family)
    n_sample = len(sample)
    nStim = n_type * n_family * n_sample

    assert np.sum(np.tile(sample, n_type * n_family) ==
                  activations.sample.values) == nStim
    assert np.sum(np.tile(np.repeat(family, n_sample), n_type) ==
                  activations.family.values) == nStim
    assert np.sum(np.repeat(type, n_family * n_sample) ==
                  activations.type.values) == nStim


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
    pref_or_fit = np.argmax(or_curve_full[180:360])
    or_curve_max = or_curve_full[pref_or_fit + 180]

    try:
        less = np.where(or_curve_full <= or_curve_max * thrsh)[0][:]
        p1 = or_full[less[np.where(less < pref_or_fit + 180)[0][-1]]]
        p2 = or_full[less[np.where(less > pref_or_fit + 180)[0][0]]]
        bw = (p2 - p1)
        if bw > 180:
            bw = np.nan
    except:
        bw = np.nan
    if mode is 'half':
        bw = bw / 2
    return bw, pref_or_fit, or_full[180:360], or_curve_full[180:360]


def calc_orthogonal_preferred_ratio(orientation_curve, orientation):
    pref_orientation = np.argmax(orientation_curve)
    orth_orientation = pref_orientation + int(len(orientation) / 2)
    if orth_orientation >= len(orientation):
        orth_orientation -= len(orientation)
    opr = orientation_curve[orth_orientation] / orientation_curve[pref_orientation]
    return opr


def calc_spatial_frequency_tuning(y, sf, filt_type='triangle', thrsh=0.707, mode='ratio'):
    from scipy.interpolate import UnivariateSpline
    sf_log = np.log2(sf)
    sf_values = y
    sf_log_full = np.linspace(sf_log[0], sf_log[-1], num=100, endpoint=True)

    if filt_type == 'hanning':
        w = np.array([0, 2/5, 1, 2/5, 0])
    elif filt_type == 'flat':
        w = np.array([1, 1, 1, 1, 1])
    elif filt_type == 'smooth':
        w = np.array([0, 1/5, 1, 1/5, 0])
    elif filt_type == 'triangle':
        w = np.array([0.5, 0.75, 1, 0.75, 0.5])

    if filt_type is not None:
        sf_values = np.convolve(w / w.sum(), np.squeeze(np.concatenate((np.array([sf_values[0], sf_values[0]]),
                                                                        sf_values, np.array([sf_values[-1],
                                                                                             sf_values[-1]])))),
                                mode='valid')
    sf_curve_spl = UnivariateSpline(sf_log, sf_values, s=0.)

    sf_curve_full = sf_curve_spl(sf_log_full)

    pref_sf_fit = np.argmax(sf_curve_full)
    sf_pk_log = sf_log_full[pref_sf_fit]

    sf_curve_max = sf_curve_full[pref_sf_fit]
    less = np.where(sf_curve_full <= sf_curve_max * thrsh)[0][:]

    try:
        p1_log = sf_log_full[less[np.where(less < pref_sf_fit)[0][-1]]]
    except:
        p1_log = np.nan
    try:
        p2_log = sf_log_full[less[np.where(less > pref_sf_fit)[0][0]]]
    except:
        p2_log = np.nan

    if mode == 'oct':
        bw = (2 ** p2_log) / (2 ** p1_log) - 1
    else:
        bw = (2 ** p1_log) / (2 ** p2_log) * 100

    values_fitted = sf_curve_spl(np.log2(sf))
    ss_res = np.sum((y - values_fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return bw, np.power(2, sf_pk_log), r2, np.power(2, sf_log_full), sf_curve_full


def calc_size_tuning(size_curve, radius):
    pref_rad = np.argmax(size_curve)
    surr_peak_r = np.max(size_curve)
    surr_plateau_r = size_curve[-1]
    ssi = (surr_peak_r - surr_plateau_r) / surr_peak_r
    if surr_peak_r > 0 and ssi > 0.1:
        gsf = radius[np.where(size_curve >= (surr_peak_r * 0.95))[0][0]] * 2
        thrsh = surr_plateau_r + 0.05 * np.absolute(surr_plateau_r)
        surr_diam = radius[np.where(np.logical_and(size_curve <= thrsh, radius > radius[pref_rad]))[0][0]] * 2
        surr_gsf_ratio = surr_diam / gsf
    else:
        gsf, surr_diam, surr_gsf_ratio = np.nan, np.nan, np.nan

    return gsf, surr_diam, surr_gsf_ratio, ssi


def calc_texture_modulation(response):
    texture_modulation_family = (response[1, :] - response[0, :]) / (response[1, :] + response[0, :])
    texture_modulation = np.nanmean(texture_modulation_family)
    return texture_modulation, texture_modulation_family


def calc_sparseness(response):
    response = response.reshape(-1)
    n_stim = response.shape[0]
    sparseness = (1 - ((response.sum() / n_stim) ** 2) / ((response ** 2).sum() / n_stim)) / (1 - 1 / n_stim)
    return sparseness


def calc_variance_ratio(response):
    residual_ms, sample_ms, family_ms = calc_variance(response)
    response_shape = response.shape
    if len(response_shape) == 3:
        residual_variance = residual_ms
        sample_variance = (sample_ms - residual_ms) / response_shape[2]
        family_variance = (family_ms - sample_ms) / (response_shape[2]*response_shape[1])
    else:
        residual_variance = 0
        sample_variance = sample_ms
        family_variance = (family_ms - sample_ms) / response_shape[1]
    total_variance = residual_variance + sample_variance + family_variance
    variance_ratio = (family_variance / total_variance + 0.02) / (sample_variance / total_variance + 0.02)
    return variance_ratio, sample_variance / total_variance, family_variance / total_variance


def calc_variance(response):
    response_shape = response.shape
    if len(response_shape) == 3:
        a, b, n = response_shape
        sample_mean = response.mean(axis=2)
        family_mean = sample_mean.mean(axis=1)
        all_mean = family_mean.mean()
        residual_ms = np.sum((response - sample_mean.reshape(a, b, 1)) ** 2) / (a * b * (n - 1))
        sample_ms = n * np.sum((sample_mean - family_mean.reshape(a, 1)) ** 2) / (a * (b - 1))
        family_ms = b*n*np.sum((family_mean - all_mean) ** 2) / (a - 1)
    else:
        a, b = response_shape
        sample_mean = response
        family_mean = sample_mean.mean(axis=1)
        all_mean = family_mean.mean()
        residual_ms = np.nan
        sample_ms = np.sum((sample_mean - family_mean.reshape(a, 1)) ** 2) / (a * (b - 1))
        family_ms = b * np.sum((family_mean - all_mean) ** 2) / (a - 1)
    return residual_ms, sample_ms, family_ms
