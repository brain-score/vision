from pathlib import Path

import pytest
from pytest import approx

from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, PrecomputedTests
from brainscore_vision.data_helpers import s3

standardized_tests = StandardizedTests()
precomputed_test = PrecomputedTests()


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('Rajalingham2020.IT-pls', approx(.561013, abs=.001),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('Rajalingham2020.IT-pls', 8, approx(.693463, abs=.005),
                 marks=[pytest.mark.memory_intense, pytest.mark.slow]),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.slow
@pytest.mark.parametrize('benchmark, expected', [
    ('Rajalingham2020.IT-pls', approx(.147549, abs=.01)),
])
def test_Rajalingham2020(benchmark, expected):
    filename = 'alexnet-rajalingham2020-features.12.nc'
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(local_path=filepath,
                                   bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_test.run_test(benchmark=benchmark, precomputed_features_filepath=filepath, expected=expected)
