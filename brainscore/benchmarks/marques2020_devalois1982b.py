import numpy as np

import brainscore
from brainio_base.assemblies import DataAssembly
from brainscore.benchmarks._properties_common import PropertiesBenchmark, _assert_grating_activations
from brainscore.benchmarks._properties_common import calc_spatial_frequency_tuning
from brainscore.metrics.ceiling import NeuronalPropertyCeiling
from brainscore.metrics.distribution_similarity import BootstrapDistributionSimilarity, ks_similarity
from result_caching import store

ASSEMBLY_NAME = 'devalois.DeValois1982b'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT = 'V1-spatial_frequency'

PROPERTY_NAMES = ['peak_spatial_frequency']

BIBTEX = """@article{DeValois1982,
            author = {{De Valois}, Russell L. and Albrecht, Duane G. and Thorell, Lisa G.},
            journal = {Vision Research},
            pages = {545--559},
            title = {{Spatial Frequency Selectivity of Cells in Macaque Visual Cortex}},
            volume = {22},
            year = {1982}
            }
            """

RESPONSE_THRESHOLD = 5


def MarquesDeValois1982V1PeakSpatialFrequency():
    assembly = brainscore.get_assembly(ASSEMBLY_NAME)
    property_name = 'peak_spatial_frequency'
    parent = PARENT
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_similarity, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_similarity,
                                                                        property_name=property_name))
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_devalois1982-{property_name}', assembly=assembly,
                               neuronal_property=devalois1982b_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


@store(identifier_ignore=['responses', 'baseline'])
def devalois1982b_properties(model_identifier, responses, baseline):
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

    peak_spatial_frequency = np.zeros((n_neuroids, 1))

    for neur in range(n_neuroids):
        pref_radius, pref_spatial_frequency, pref_orientation, pref_component = \
            np.unravel_index(np.argmax(responses[neur, :, :, :, :]),
                             (len(radius), len(spatial_frequency), len(orientation), 2))

        spatial_frequency_curve = responses[neur, pref_radius, :, pref_orientation, pref_component]

        peak_spatial_frequency[neur] = \
            calc_spatial_frequency_tuning(spatial_frequency_curve, spatial_frequency, thrsh=0.707, filt_type='smooth',
                                          mode='ratio')[1]

    properties_data = peak_spatial_frequency

    good_neuroids = max_response > RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
