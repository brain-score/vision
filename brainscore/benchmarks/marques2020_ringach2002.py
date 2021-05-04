import numpy as np

import brainscore
from brainio_base.assemblies import DataAssembly
from brainscore.benchmarks._properties_common import PropertiesBenchmark, _assert_grating_activations
from brainscore.benchmarks._properties_common import calc_circular_variance, calc_bandwidth, \
    calc_orthogonal_preferred_ratio
from brainscore.metrics.ceiling import NeuronalPropertyCeiling
from brainscore.metrics.distribution_similarity import BootstrapDistributionSimilarity, ks_similarity
from result_caching import store

ASSEMBLY_NAME = 'shapley.Ringach2002'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT_ORIENTATION = 'V1-orientation'
PARENT_MAGNITUDE = 'V1-response_magnitude'
PARENT_SELECTIVITY = 'V1-response_selectivity'

PROPERTY_NAMES = ['baseline', 'max_dc', 'min_dc', 'max_ac', 'modulation_ratio', 'circular_variance', 'bandwidth',
                  'orthogonal_preferred_ratio', 'orientation_selective', 'circular_variance_bandwidth_ratio',
                  'orthogonal_preferred_ratio_circular_variance_difference',
                  'orthogonal_preferred_ratio_bandwidth_ratio']

BIBTEX = """@article{Ringach2002,
            author = {Ringach, Dario L and Shapley, Robert M and Hawken, Michael J},
            doi = {20026567},
            issn = {1529-2401},
            journal = {The Journal of Neuroscience},
            number = {13},
            pages = {5639--5651},
            pmid = {12097515},
            title = {{Orientation selectivity in macaque V1: diversity and laminar dependence.}},
            volume = {22},
            year = {2002}
            }"""

RESPONSE_THRESHOLD = 5


def _MarquesRingach2002V1Property(property_name, parent):
    assembly = brainscore.get_assembly(ASSEMBLY_NAME)
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_similarity, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_similarity,
                                                                        property_name=property_name))
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_ringach2002-{property_name}', assembly=assembly,
                               neuronal_property=ringach2002_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


def MarquesRingach2002V1CircularVariance():
    property_name = 'circular_variance'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1Bandwidth():
    property_name = 'bandwidth'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1OrthogonalPreferredRatio():
    property_name = 'orthogonal_preferred_ratio'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1OrientationSelective():
    property_name = 'orientation_selective'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1CircularVarianceBandwidthRatio():
    property_name = 'circular_variance_bandwidth_ratio'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference():
    property_name = 'orthogonal_preferred_ratio_circular_variance_difference'
    parent = PARENT_ORIENTATION
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1MaxDC():
    property_name = 'max_dc'
    parent = PARENT_MAGNITUDE
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


def MarquesRingach2002V1ModulationRatio():
    property_name = 'modulation_ratio'
    parent = PARENT_SELECTIVITY
    return _MarquesRingach2002V1Property(property_name=property_name, parent=parent)


@store(identifier_ignore=['responses', 'baseline'])
def ringach2002_properties(model_identifier, responses, baseline):
    _assert_grating_activations(responses)
    spatial_frequency = np.array(sorted(set(responses.spatial_frequency.values)))
    orientation = np.array(sorted(set(responses.orientation.values)))
    phase = np.array(sorted(set(responses.phase.values)))
    nStim = responses.values.shape[1]
    n_cycles = nStim // (len(phase) * len(orientation) * len(spatial_frequency))

    responses = responses.values
    baseline = baseline.values
    assert responses.shape[0] == baseline.shape[0]
    n_neuroids = responses.shape[0]

    responses = responses.reshape((n_neuroids, n_cycles, len(spatial_frequency), len(orientation), len(phase)))
    responses_dc = responses.mean(axis=4)
    responses_ac = np.absolute(np.fft.fft(responses)) / len(phase)
    responses_ac = responses_ac[:, :, :, :, 1] * 2
    del responses

    max_dc = np.zeros((n_neuroids, 1))
    max_ac = np.zeros((n_neuroids, 1))
    min_dc = np.zeros((n_neuroids, 1))
    circular_variance = np.zeros((n_neuroids, 1))
    bandwidth = np.zeros((n_neuroids, 1))
    orthogonal_preferred_ratio = np.zeros((n_neuroids, 1))
    orientation_selective = np.ones((n_neuroids, 1))

    for neur in range(n_neuroids):
        pref_cycle, pref_spatial_frequency, pref_orientation = np.unravel_index(np.argmax(responses_dc[neur]),
                                                                                (n_cycles, len(spatial_frequency),
                                                                                 len(orientation)))

        max_dc[neur] = responses_dc[neur, pref_cycle, pref_spatial_frequency, pref_orientation]
        max_ac[neur] = responses_ac[neur, pref_cycle, pref_spatial_frequency, pref_orientation]

        orientation_curve = responses_dc[neur, pref_cycle, pref_spatial_frequency, :]
        min_dc[neur] = orientation_curve.min()

        circular_variance[neur] = calc_circular_variance(orientation_curve, orientation)
        bandwidth[neur] = \
            calc_bandwidth(orientation_curve, orientation, filt_type='hanning', thrsh=0.707, mode='half')[0]
        orthogonal_preferred_ratio[neur] = calc_orthogonal_preferred_ratio(orientation_curve, orientation)

    orientation_selective[np.isnan(bandwidth)] = 0
    modulation_ratio = max_ac / max_dc
    circular_variance_bandwidth_ratio = circular_variance / bandwidth
    orthogonal_preferred_ratio_circular_variance_difference = orthogonal_preferred_ratio - circular_variance
    orthogonal_preferred_ratio_bandwidth_ratio = orthogonal_preferred_ratio / bandwidth

    properties_data = np.concatenate(
        (baseline, max_dc, min_dc, max_ac, modulation_ratio, circular_variance, bandwidth, orthogonal_preferred_ratio,
         orientation_selective, circular_variance_bandwidth_ratio,
         orthogonal_preferred_ratio_circular_variance_difference, orthogonal_preferred_ratio_bandwidth_ratio), axis=1)

    good_neuroids = max_dc > baseline + RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
