import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials, \
    TestBenchmarkRegistry

# should these be in function definitions
standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()
registry_test = TestBenchmarkRegistry()


@pytest.mark.parametrize('benchmark', [
        'dicarlo.Marques2020_Cavanaugh2002-grating_summation_field',
        'dicarlo.Marques2020_Cavanaugh2002-surround_diameter',
        'dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index',
    ])
def test_benchmark_registry(benchmark):
    registry_test.test_benchmark_registry(benchmark)

@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Marques2020_Cavanaugh2002-grating_summation_field', approx(0.956, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_Cavanaugh2002-surround_diameter', approx(0.955, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index', approx(0.958, abs=.005), marks=[]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)

@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Marques2020_Cavanaugh2002-grating_summation_field', approx(.599, abs=.01)),
    ('dicarlo.Marques2020_Cavanaugh2002-surround_diameter', approx(.367, abs=.01)),
    ('dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index', approx(.365, abs=.01)),
])
def test_Marques2020(benchmark, expected):
    precomputed_test.run_test_properties(
        benchmark=benchmark,
        files={'dicarlo.Marques2020_blank': 'alexnet-dicarlo.Marques2020_blank.nc',
               'dicarlo.Marques2020_receptive_field': 'alexnet-dicarlo.Marques2020_receptive_field.nc',
               'dicarlo.Marques2020_orientation': 'alexnet-dicarlo.Marques2020_orientation.nc',
               'dicarlo.Marques2020_spatial_frequency': 'alexnet-dicarlo.Marques2020_spatial_frequency.nc',
               'dicarlo.Marques2020_size': 'alexnet-dicarlo.Marques2020_size.nc',
               'movshon.FreemanZiemba2013_properties': 'alexnet-movshon.FreemanZiemba2013_properties.nc',
               },
        expected=expected)

@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.Marques2020_Cavanaugh2002-grating_summation_field',
    'dicarlo.Marques2020_Cavanaugh2002-surround_diameter',
    'dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)