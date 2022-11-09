import numpy as np

import brainscore_vision
from brainio.assemblies import DataAssembly
from brainscore_vision.benchmarks._properties_common import PropertiesBenchmark, _assert_grating_activations
from brainscore_vision.benchmarks._properties_common import calc_spatial_frequency_tuning
from brainscore_vision.metrics.ceiling import NeuronalPropertyCeiling
from brainscore_vision.metrics.distribution_similarity import BootstrapDistributionSimilarity, ks_similarity
from result_caching import store

ASSEMBLY_NAME = 'schiller.Schiller1976c'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT = 'V1-spatial_frequency'

PROPERTY_NAMES = ['spatial_frequency_selective', 'spatial_frequency_bandwidth']

BIBTEX = """@article{Schiller1976,
            author = {Schiller, P. H. and Finlay, B. L. and Volman, S. F.},
            doi = {10.1152/jn.1976.39.6.1352},
            issn = {0022-3077},
            journal = {Journal of neurophysiology},
            number = {6},
            pages = {1334--1351},
            pmid = {825624},
            title = {{Quantitative studies of single-cell properties in monkey striate cortex. III. Spatial Frequency}},
            url = {http://www.ncbi.nlm.nih.gov/pubmed/825624},
            volume = {39},
            year = {1976}
            }
            """

RESPONSE_THRESHOLD = 5


def _MarquesSchiller1976V1Property(property_name):
    assembly = brainscore_vision.get_assembly(ASSEMBLY_NAME)
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_similarity, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(similarity_metric)
    parent = PARENT
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_schiller1976-{property_name}', assembly=assembly,
                               neuronal_property=schiller1976_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


def MarquesSchiller1976V1SpatialFrequencySelective():
    property_name = 'spatial_frequency_selective'
    return _MarquesSchiller1976V1Property(property_name=property_name)


def MarquesSchiller1976V1SpatialFrequencyBandwidth():
    property_name = 'spatial_frequency_bandwidth'
    return _MarquesSchiller1976V1Property(property_name=property_name)


@store(identifier_ignore=['responses', 'baseline'])
def schiller1976_properties(model_identifier, responses, baseline):
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
    responses = responses.mean(axis=4)

    max_response = responses.reshape((n_neuroids, -1)).max(axis=1, keepdims=True)

    spatial_frequency_bandwidth = np.zeros((n_neuroids, 1))
    spatial_frequency_selective = np.ones((n_neuroids, 1))

    for neur in range(n_neuroids):
        pref_radius, pref_spatial_frequency, pref_orientation = \
            np.unravel_index(np.argmax(responses[neur, :, :, :]),
                             (len(radius), len(spatial_frequency), len(orientation)))

        spatial_frequency_curve = responses[neur, pref_radius, :, pref_orientation]

        spatial_frequency_bandwidth[neur] = \
            calc_spatial_frequency_tuning(spatial_frequency_curve, spatial_frequency, thrsh=0.707, filt_type='smooth',
                                          mode='ratio')[0]

    spatial_frequency_selective[np.isnan(spatial_frequency_bandwidth)] = 0

    properties_data = np.concatenate((spatial_frequency_selective, spatial_frequency_bandwidth), axis=1)

    good_neuroids = max_response > baseline + RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
