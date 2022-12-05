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
    'dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index',
    'dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
    'dicarlo.Marques2020_FreemanZiemba2013-max_noise',
    'dicarlo.Marques2020_FreemanZiemba2013-max_texture',
])
def test_benchmark_registry(benchmark):
    registry_test.test_benchmark_registry(benchmark)


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index', approx(0.948, abs=.005),
                 marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(0.958, abs=.005),
                 marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity', approx(0.940, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness', approx(0.935, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(0.939, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-max_texture', approx(0.946, abs=.005), marks=[]),
    pytest.param('dicarlo.Marques2020_FreemanZiemba2013-max_noise', approx(0.945, abs=.005), marks=[]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index', approx(.636, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(.861, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity', approx(.646, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness', approx(.508, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(.827, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-max_texture', approx(.823, abs=.01)),
    ('dicarlo.Marques2020_FreemanZiemba2013-max_noise', approx(.684, abs=.01)),
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
    'dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio',
    'dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index',
    'dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
    'dicarlo.Marques2020_FreemanZiemba2013-max_noise',
    'dicarlo.Marques2020_FreemanZiemba2013-max_texture',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)
