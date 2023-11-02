import numpy as np
from result_caching import store

from brainio.assemblies import DataAssembly
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.properties_common import PropertiesBenchmark, _assert_texture_activations
from brainscore_vision.benchmark_helpers.properties_common import calc_texture_modulation, calc_sparseness, \
    calc_variance_ratio
from brainscore_vision.data.freemanziemba2013 import BIBTEX
from brainscore_vision.metrics.distribution_similarity import NeuronalPropertyCeiling

ASSEMBLY_NAME = 'movshon.FreemanZiemba2013_V1_properties'
REGION = 'V1'
TIMEBINS = [(70, 170)]
PARENT_TEXTURE_MODULATION = 'V1-texture_modulation'
PARENT_SELECTIVITY = 'V1-response_selectivity'
PARENT_MAGNITUDE = 'V1-response_magnitude'

PROPERTY_NAMES = ['texture_modulation_index', 'absolute_texture_modulation_index', 'texture_selectivity',
                  'noise_selectivity', 'texture_sparseness', 'noise_sparseness', 'variance_ratio', 'sample_variance',
                  'family_variance', 'max_texture', 'max_noise']

RESPONSE_THRESHOLD = 5


def _MarquesFreemanZiemba2013V1Property(property_name, parent):
    assembly = load_dataset(ASSEMBLY_NAME)
    similarity_metric = load_metric('ks_similarity', property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(similarity_metric)
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_freemanziemba2013-{property_name}', assembly=assembly,
                               neuronal_property=freemanziemba2013_properties, similarity_metric=similarity_metric,
                               timebins=TIMEBINS,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=1)


def MarquesFreemanZiemba2013V1TextureModulationIndex():
    property_name = 'texture_modulation_index'
    parent = PARENT_TEXTURE_MODULATION
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex():
    property_name = 'absolute_texture_modulation_index'
    parent = PARENT_TEXTURE_MODULATION
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1TextureSelectivity():
    property_name = 'texture_selectivity'
    parent = PARENT_SELECTIVITY
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1TextureSparseness():
    property_name = 'texture_sparseness'
    parent = PARENT_SELECTIVITY
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1VarianceRatio():
    property_name = 'variance_ratio'
    parent = PARENT_SELECTIVITY
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1MaxTexture():
    property_name = 'max_texture'
    parent = PARENT_MAGNITUDE
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


def MarquesFreemanZiemba2013V1MaxNoise():
    property_name = 'max_noise'
    parent = PARENT_MAGNITUDE
    return _MarquesFreemanZiemba2013V1Property(property_name=property_name, parent=parent)


@store(identifier_ignore=['responses', 'baseline'])
def freemanziemba2013_properties(model_identifier, responses, baseline):
    _assert_texture_activations(responses)
    responses = responses.sortby(['type', 'family', 'sample'])
    type = np.array(sorted(set(responses.type.values)))
    family = np.array(sorted(set(responses.family.values)))
    sample = np.array(sorted(set(responses.sample.values)))

    responses = responses.values
    baseline = baseline.values
    assert responses.shape[0] == baseline.shape[0]
    n_neuroids = responses.shape[0]

    responses = responses.reshape(n_neuroids, len(type), len(family), len(sample))
    responses_spikes = responses / 10
    responses_spikes = np.sqrt(responses_spikes) + np.sqrt(responses_spikes + 1)
    responses -= baseline.reshape((-1, 1, 1, 1))

    max_texture = np.max((responses.reshape((n_neuroids, 2, -1)))[:, 1, :], axis=1, keepdims=True)
    max_noise = np.max((responses.reshape((n_neuroids, 2, -1)))[:, 0, :], axis=1, keepdims=True)
    max_response = np.max(responses.reshape((n_neuroids, -1)), axis=1, keepdims=True)

    responses_family = responses.mean(axis=3)

    texture_modulation_index = np.zeros((n_neuroids, 1))
    texture_selectivity = np.zeros((n_neuroids, 1))
    noise_selectivity = np.zeros((n_neuroids, 1))
    texture_sparseness = np.zeros((n_neuroids, 1))
    noise_sparseness = np.zeros((n_neuroids, 1))
    variance_ratio = np.zeros((n_neuroids, 1))
    sample_variance = np.zeros((n_neuroids, 1))
    family_variance = np.zeros((n_neuroids, 1))

    for neur in range(n_neuroids):
        texture_modulation_index[neur] = calc_texture_modulation(responses_family[neur])[0]
        texture_selectivity[neur] = calc_sparseness(responses_family[neur, 1])
        noise_selectivity[neur] = calc_sparseness(responses_family[neur, 0])
        texture_sparseness[neur] = calc_sparseness(responses[neur, 1])
        noise_sparseness[neur] = calc_sparseness(responses[neur, 0])
        variance_ratio[neur], sample_variance[neur], family_variance[neur] = \
            calc_variance_ratio(responses_spikes[neur, 1])

    absolute_texture_modulation_index = np.abs(texture_modulation_index)

    properties_data = np.concatenate((texture_modulation_index, absolute_texture_modulation_index, texture_selectivity,
                                      noise_selectivity, texture_sparseness, noise_sparseness, variance_ratio,
                                      sample_variance, family_variance, max_texture, max_noise), axis=1)

    good_neuroids = max_response > RESPONSE_THRESHOLD
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data
