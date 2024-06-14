import pytest
from pytest import approx

from brainscore_vision import benchmark_registry
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, PrecomputedTests, NumberOfTrialsTests

standardized_tests = StandardizedTests()
precomputed_test = PrecomputedTests()
num_trials_test = NumberOfTrialsTests()


@pytest.mark.parametrize('benchmark', [
    'Marques2020_Cavanaugh2002-grating_summation_field',
    'Marques2020_Cavanaugh2002-surround_diameter',
    'Marques2020_Cavanaugh2002-surround_suppression_index',
    'Marques2020_DeValois1982-pref_or',
    'Marques2020_DeValois1982-peak_sf',
    'Marques2020_FreemanZiemba2013-texture_sparseness',
    'Marques2020_FreemanZiemba2013-texture_selectivity',
    'Marques2020_FreemanZiemba2013-texture_variance_ratio',
    'Marques2020_FreemanZiemba2013-texture_modulation_index',
    'Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
    'Marques2020_FreemanZiemba2013-max_noise',
    'Marques2020_FreemanZiemba2013-max_texture',
    'Marques2020_Ringach2002-or_bandwidth',
    'Marques2020_Ringach2002-or_selective',
    'Marques2020_Ringach2002-circular_variance',
    'Marques2020_Ringach2002-orth_pref_ratio',
    'Marques2020_Ringach2002-cv_bandwidth_ratio',
    'Marques2020_Ringach2002-opr_cv_diff',
    'Marques2020_Ringach2002-modulation_ratio',
    'Marques2020_Ringach2002-max_dc',
    'Marques2020_Schiller1976-sf_bandwidth',
    'Marques2020_Schiller1976-sf_selective',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark, expected', [
    ('Marques2020_Cavanaugh2002-grating_summation_field', approx(0.956, abs=.005)),
    ('Marques2020_Cavanaugh2002-surround_diameter', approx(0.955, abs=.005)),
    ('Marques2020_Cavanaugh2002-surround_suppression_index', approx(0.958, abs=.005)),
    ('Marques2020_DeValois1982-pref_or', approx(0.962, abs=.005)),
    ('Marques2020_DeValois1982-peak_sf', approx(0.967, abs=.005)),
    ('Marques2020_FreemanZiemba2013-texture_modulation_index', approx(0.948, abs=.005)),
    ('Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(0.958, abs=.005)),
    ('Marques2020_FreemanZiemba2013-texture_selectivity', approx(0.940, abs=.005)),
    ('Marques2020_FreemanZiemba2013-texture_sparseness', approx(0.935, abs=.005)),
    ('Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(0.939, abs=.005)),
    ('Marques2020_FreemanZiemba2013-max_texture', approx(0.946, abs=.005)),
    ('Marques2020_FreemanZiemba2013-max_noise', approx(0.945, abs=.005)),
    ('Marques2020_Ringach2002-circular_variance', approx(0.959, abs=.005)),
    ('Marques2020_Ringach2002-or_bandwidth', approx(0.964, abs=.005)),
    ('Marques2020_Ringach2002-orth_pref_ratio', approx(0.962, abs=.005)),
    ('Marques2020_Ringach2002-or_selective', approx(0.994, abs=.005)),
    ('Marques2020_Ringach2002-cv_bandwidth_ratio', approx(0.968, abs=.005)),
    ('Marques2020_Ringach2002-opr_cv_diff', approx(0.967, abs=.005)),
    ('Marques2020_Ringach2002-modulation_ratio', approx(0.959, abs=.005)),
    ('Marques2020_Ringach2002-max_dc', approx(0.968, abs=.005)),
    ('Marques2020_Schiller1976-sf_selective', approx(0.963, abs=.005)),
    ('Marques2020_Schiller1976-sf_bandwidth', approx(0.933, abs=.005)),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('Marques2020_Cavanaugh2002-grating_summation_field', approx(.599, abs=.01)),
    ('Marques2020_Cavanaugh2002-surround_diameter', approx(.367, abs=.01)),
    ('Marques2020_Cavanaugh2002-surround_suppression_index', approx(.365, abs=.01)),
    ('Marques2020_DeValois1982-pref_or', approx(.895, abs=.01)),
    ('Marques2020_DeValois1982-peak_sf', approx(.775, abs=.01)),
    ('Marques2020_FreemanZiemba2013-texture_modulation_index', approx(.636, abs=.01)),
    ('Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(.861, abs=.01)),
    ('Marques2020_FreemanZiemba2013-texture_selectivity', approx(.646, abs=.01)),
    ('Marques2020_FreemanZiemba2013-texture_sparseness', approx(.508, abs=.01)),
    ('Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(.827, abs=.01)),
    ('Marques2020_FreemanZiemba2013-max_texture', approx(.823, abs=.01)),
    ('Marques2020_FreemanZiemba2013-max_noise', approx(.684, abs=.01)),
    ('Marques2020_Ringach2002-circular_variance', approx(.830, abs=.01)),
    ('Marques2020_Ringach2002-or_bandwidth', approx(.844, abs=.01)),
    ('Marques2020_Ringach2002-orth_pref_ratio', approx(.876, abs=.01)),
    ('Marques2020_Ringach2002-or_selective', approx(.895, abs=.01)),
    ('Marques2020_Ringach2002-cv_bandwidth_ratio', approx(.841, abs=.01)),
    ('Marques2020_Ringach2002-opr_cv_diff', approx(.909, abs=.01)),
    ('Marques2020_Ringach2002-modulation_ratio', approx(.371, abs=.01)),
    ('Marques2020_Ringach2002-max_dc', approx(.904, abs=.01)),
    ('Marques2020_Schiller1976-sf_selective', approx(.808, abs=.01)),
    ('Marques2020_Schiller1976-sf_bandwidth', approx(.869, abs=.01)),
])
def test_Marques2020(benchmark, expected):
    precomputed_test.run_test_properties(
        benchmark=benchmark,
        files={'Marques2020_blank': 'alexnet-dicarlo.Marques2020_blank.nc',
               'Marques2020_receptive_field': 'alexnet-dicarlo.Marques2020_receptive_field.nc',
               'Marques2020_orientation': 'alexnet-dicarlo.Marques2020_orientation.nc',
               'Marques2020_spatial_frequency': 'alexnet-dicarlo.Marques2020_spatial_frequency.nc',
               'Marques2020_size': 'alexnet-dicarlo.Marques2020_size.nc',
               'FreemanZiemba2013_properties': 'alexnet-movshon.FreemanZiemba2013_properties.nc',
               },
        expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'Marques2020_Cavanaugh2002-grating_summation_field',
    'Marques2020_Cavanaugh2002-surround_diameter',
    'Marques2020_Cavanaugh2002-surround_suppression_index',
    'Marques2020_DeValois1982-pref_or',
    'Marques2020_DeValois1982-peak_sf',
    'Marques2020_FreemanZiemba2013-texture_sparseness',
    'Marques2020_FreemanZiemba2013-texture_selectivity',
    'Marques2020_FreemanZiemba2013-texture_variance_ratio',
    'Marques2020_FreemanZiemba2013-texture_modulation_index',
    'Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
    'Marques2020_FreemanZiemba2013-max_noise',
    'Marques2020_FreemanZiemba2013-max_texture',
    'Marques2020_Ringach2002-or_bandwidth',
    'Marques2020_Ringach2002-or_selective',
    'Marques2020_Ringach2002-circular_variance',
    'Marques2020_Ringach2002-orth_pref_ratio',
    'Marques2020_Ringach2002-cv_bandwidth_ratio',
    'Marques2020_Ringach2002-opr_cv_diff',
    'Marques2020_Ringach2002-modulation_ratio',
    'Marques2020_Ringach2002-max_dc',
    'Marques2020_Schiller1976-sf_bandwidth',
    'Marques2020_Schiller1976-sf_selective',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)
