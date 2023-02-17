import pytest
from pytest import approx
from brainscore_vision.benchmark_helpers.test_helper import TestStandardized, TestPrecomputed


standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('dicarlo.Rajalingham2020.IT-pls', approx(.561013, abs=.001),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('dicarlo.Rajalingham2020.IT-pls', 8, approx(.693463, abs=.005),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('dicarlo.Rajalingham2020.IT-pls', approx(.147549, abs=.01)),
])
def test_Rajalingham2020(benchmark, expected):
    precomputed_test.run_test(benchmark=benchmark, file='alexnet-rajalingham2020-features.12.nc', expected=expected)

