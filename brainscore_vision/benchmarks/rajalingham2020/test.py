import pytest
from pytest import approx
from brainscore_vision.benchmarks.test_helper import TestStandardized, TestPrecomputed


standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Rajalingham2020.IT-pls', approx(.561013, abs=.001),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.test_ceilings(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.Rajalingham2020.IT-pls', 8, approx(.693463, abs=.005),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.test_self_regression(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Rajalingham2020.IT-pls', approx(.147549, abs=.01)),
])
def test_Rajalingham2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-rajalingham2020-features.12.nc', expected=expected)

