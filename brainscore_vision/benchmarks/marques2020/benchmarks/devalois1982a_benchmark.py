import numpy as np
from result_caching import store

from brainio.assemblies import DataAssembly
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.properties_common import PropertiesBenchmark, _assert_grating_activations
from brainscore_vision.benchmark_helpers.properties_common import calc_bandwidth
from brainscore_vision.metrics.distribution_similarity import NeuronalPropertyCeiling

ASSEMBLY_NAME = 'devalois.DeValois1982a'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT = 'V1-orientation'

PROPERTY_NAMES = ['preferred_orientation']

BIBTEX = """@article{DeValois1982,
            author = {{De Valois}, Russell L. and Yund, E. W. and Hepler, Norva},
            journal = {Vision Research},
            pages = {531--544},
            title = {{The orientation and direction selectivity of cells in macaque visual cortex}},
            volume = {22},
            year = {1982}
            }
            """

RESPONSE_THRESHOLD = 5
ORIENTATION_BIN_LIM = 157.5


def MarquesDeValois1982V1PreferredOrientation():
    assembly = load_dataset(ASSEMBLY_NAME)
    property_name = 'preferred_orientation'
    parent = PARENT
    similarity_metric = load_metric('ks_similarity', property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(similarity_metric)
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_devalois1982-{property_name}', assembly=assembly,
                               neuronal_property=devalois1982a_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


@store(identifier_ignore=['responses', 'baseline'])
def devalois1982a_properties(model_identifier, responses, baseline):
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
    responses = responses.mean(axis=4)

    preferred_orientation = np.zeros((n_neuroids, 1))
    max_response = responses.reshape((n_neuroids, -1)).max(axis=1, keepdims=True)

    for neur in range(n_neuroids):
        pref_cycle, pref_spatial_frequency, pref_orientation = np.unravel_index(np.argmax(responses[neur]),
                                                                                (n_cycles, len(spatial_frequency),
                                                                                 len(orientation)))

        orientation_curve = responses[neur, pref_cycle, pref_spatial_frequency, :]

        preferred_orientation[neur] = \
            calc_bandwidth(orientation_curve, orientation, filt_type='smooth', thrsh=0.5, mode='full')[1]

    preferred_orientation[preferred_orientation >= ORIENTATION_BIN_LIM] = \
        preferred_orientation[preferred_orientation >= ORIENTATION_BIN_LIM] - 180
    properties_data = preferred_orientation

    good_neuroids = max_response > baseline + RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
