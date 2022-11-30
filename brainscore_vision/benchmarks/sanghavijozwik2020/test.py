import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials

standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.SanghaviJozwik2020.V4-pls', approx(.9630336, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.SanghaviJozwik2020.IT-pls', approx(.860352, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.SanghaviJozwik2020.V4-pls', 8, approx(.9739177, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.SanghaviJozwik2020.IT-pls', 8, approx(.9999779, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.test_self_regression(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.SanghaviJozwik2020.V4-pls', approx(.49235, abs=.005)),
    ('dicarlo.SanghaviJozwik2020.IT-pls', approx(.590543, abs=.005)),
])
def test_SanghaviJozwik2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-sanghavijozwik2020-features.12.nc', expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.SanghaviJozwik2020.V4-pls',
    'dicarlo.SanghaviJozwik2020.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)