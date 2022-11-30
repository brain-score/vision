import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials

standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.SanghaviMurty2020.V4-pls', approx(.9666086, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.SanghaviMurty2020.IT-pls', approx(.875714, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.SanghaviMurty2020.V4-pls', 5, approx(.978581, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param('dicarlo.SanghaviMurty2020.IT-pls', 5, approx(.9997532, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.test_self_regression(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.SanghaviMurty2020.V4-pls', approx(.357461, abs=.015)),
    ('dicarlo.SanghaviMurty2020.IT-pls', approx(.53006, abs=.015)),
])
def test_SanghaviMurty2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-sanghavimurty2020-features.12.nc', expected=expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'dicarlo.SanghaviMurty2020.V4-pls',
    'dicarlo.SanghaviMurty2020.IT-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.test_repetitions(benchmark_identifier)
