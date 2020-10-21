import numpy as np

import brainscore
from brainio_base.assemblies import DataAssembly
from brainscore.benchmarks._properties_common import PropertiesBenchmark, _assert_grating_activations
from brainscore.benchmarks._properties_common import calc_size_tuning
from brainscore.metrics.ceiling import NeuronalPropertyCeiling
from brainscore.metrics.distribution_similarity import BootstrapDistributionSimilarity, ks_similarity
from result_caching import store

ASSEMBLY_NAME = 'movshon.Cavanaugh2002a'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT = 'V1-surround'

PROPERTY_NAMES = ['surround_suppression_index', 'strongly_suppressed', 'grating_summation_field', 'surround_diameter',
                  'surround_grating_summation_field_ratio']

BIBTEX = """@article{Cavanaugh2002,
            author = {Cavanaugh, James R. and Bair, Wyeth and Movshon, J. A.},
            doi = {10.1152/jn.00692.2001},
            isbn = {0022-3077 (Print) 0022-3077 (Linking)},
            issn = {0022-3077},
            journal = {Journal of Neurophysiology},
            mendeley-groups = {Benchmark effects/Done,Benchmark effects/*Surround Suppression},
            number = {5},
            pages = {2530--2546},
            pmid = {12424292},
            title = {{Nature and Interaction of Signals From the Receptive Field Center and Surround in Macaque V1 Neurons}},
            url = {http://www.physiology.org/doi/10.1152/jn.00692.2001},
            volume = {88},
            year = {2002}
            }
            """

RESPONSE_THRESHOLD = 5


def _MarquesCavanaugh2002V1Property(property_name):
    assembly = brainscore.get_assembly(ASSEMBLY_NAME)
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_similarity, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_similarity,
                                                                        property_name=property_name))
    parent = PARENT
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_cavanaugh2002-{property_name}', assembly=assembly,
                               neuronal_property=cavanaugh2002_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


def MarquesCavanaugh2002V1SurroundSuppressionIndex():
    property_name = 'surround_suppression_index'
    return _MarquesCavanaugh2002V1Property(property_name=property_name)


def MarquesCavanaugh2002V1GratingSummationField():
    property_name = 'grating_summation_field'
    return _MarquesCavanaugh2002V1Property(property_name=property_name)


def MarquesCavanaugh2002V1SurroundDiameter():
    property_name = 'surround_diameter'
    return _MarquesCavanaugh2002V1Property(property_name=property_name)


def MarquesCavanaugh2002V1SurroundGratingSummationFieldRatio():
    property_name = 'surround_grating_summation_field_ratio'
    return _MarquesCavanaugh2002V1Property(property_name=property_name)


@store(identifier_ignore=['responses', 'baseline'])
def cavanaugh2002_properties(model_identifier, responses, baseline):
    _assert_grating_activations(responses)
    radius = np.array(sorted(set(responses.radius.values)))
    spatial_frequency = np.array(sorted(set(responses.spatial_frequency.values)))
    orientation = np.array(sorted(set(responses.orientation.values)))
    phase = np.array(sorted(set(responses.phase.values)))

    responses = responses.values
    baseline = baseline.values
    assert responses.shape[0] == baseline.shape[0]
    n_neuroids = responses.shape[0]

    responses = responses.reshape((n_neuroids, len(radius), len(spatial_frequency), len(orientation), len(phase)))
    responses_dc = responses.mean(axis=4) - baseline.reshape((-1, 1, 1, 1))
    responses_ac = np.absolute(np.fft.fft(responses)) / len(phase)
    responses_ac = responses_ac[:, :, :, :, 1]
    responses = np.zeros((n_neuroids, len(radius), len(spatial_frequency), len(orientation), 2))
    responses[:, :, :, :, 0] = responses_dc
    responses[:, :, :, :, 1] = responses_ac
    del responses_ac, responses_dc

    max_response = responses.reshape((n_neuroids, -1)).max(axis=1, keepdims=True)

    surround_suppression_index = np.zeros((n_neuroids, 1))
    strongly_suppressed = np.zeros((n_neuroids, 1))
    grating_summation_field = np.zeros((n_neuroids, 1))
    surround_diameter = np.zeros((n_neuroids, 1))
    surround_grating_summation_field_ratio = np.zeros((n_neuroids, 1))

    for neur in range(n_neuroids):
        pref_radius, pref_spatial_frequency, pref_orientation, pref_component = \
            np.unravel_index(np.argmax(responses[neur, :, :, :, :]),
                             (len(radius), len(spatial_frequency), len(orientation), 2))

        size_curve = responses[neur, :, pref_spatial_frequency, pref_orientation, pref_component]

        grating_summation_field[neur], surround_diameter[neur], surround_grating_summation_field_ratio[neur], \
        surround_suppression_index[neur] = calc_size_tuning(size_curve, radius)

    strongly_suppressed[surround_suppression_index >= 0.1] = 1

    properties_data = np.concatenate((surround_suppression_index, strongly_suppressed, grating_summation_field,
                                      surround_diameter, surround_grating_summation_field_ratio), axis=1)

    good_neuroids = max_response > RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
