# current active tests are checking that 'dicarlo.Sanghavi2020.V4-pls' and 'dicarlo.Sanghavi2020.IT-pls'
# are in set(evaluation_benchmark_pool.keys()) TODO add these
import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials, \
    TestBenchmarkRegistry

standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()
registry_test = TestBenchmarkRegistry()

@pytest.mark.parametrize('benchmark', [
        'dicarlo.Sanghavi2020.V4-pls',
        'dicarlo.Sanghavi2020.IT-pls'
    ])
def test_benchmark_registry(benchmark):
    registry_test.test_benchmark_registry(benchmark)

@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Sanghavi2020.V4-pls', approx(.8892049, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.Sanghavi2020.IT-pls', approx(.868293, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.Sanghavi2020.V4-pls', 8, approx(.9727137, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.Sanghavi2020.IT-pls', 8, approx(.890062, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.test_self_regression(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Sanghavi2020.V4-pls', approx(.551135, abs=.015)),
    ('dicarlo.Sanghavi2020.IT-pls', approx(.611347, abs=.015)),
])
def test_Sanghavi2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-sanghavi2020-features.12.nc', expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.Sanghavi2020.V4-pls',
    'dicarlo.Sanghavi2020.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)
