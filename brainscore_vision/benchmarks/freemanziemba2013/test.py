from pathlib import Path

import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests, PrecomputedTests, NumberOfTrialsTests, \
    VisualDegreesTests
from brainscore_vision.data_helpers import s3

standardized_tests = StandardizedTests()
precomputed_test = PrecomputedTests()
num_trials_test = NumberOfTrialsTests()
visual_degrees_test = VisualDegreesTests()


@pytest.mark.parametrize('benchmark', [
    'FreemanZiemba2013.V1-pls',
    'FreemanZiemba2013.V2-pls',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, expected', [
    ('FreemanZiemba2013.V1-pls', approx(.873345, abs=.001)),
    ('FreemanZiemba2013.V2-pls', approx(.824836, abs=.001)),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    ('FreemanZiemba2013.V1-pls', 4, approx(.668491, abs=.001)),
    ('FreemanZiemba2013.V2-pls', 4, approx(.553155, abs=.001)),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, expected', [
    ('FreemanZiemba2013.V1-pls', approx(.466222, abs=.005)),
    ('FreemanZiemba2013.V2-pls', approx(.459283, abs=.005)),
])
def test_FreemanZiemba2013(benchmark, expected):
    filename = 'alexnet-freemanziemba2013.aperture-private.nc'
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(local_path=filepath,
                                   bucket='brain-score-tests', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_test.run_test(benchmark=benchmark, precomputed_features_filepath=filepath, expected=expected)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('FreemanZiemba2013.V1-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('FreemanZiemba2013.V1-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('FreemanZiemba2013public.V1-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.314561, abs=.0001), marks=[]),
    pytest.param('FreemanZiemba2013public.V1-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.23113, abs=.0001), marks=[]),
    pytest.param('FreemanZiemba2013.V2-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('FreemanZiemba2013.V2-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('FreemanZiemba2013public.V2-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.314561, abs=.0001), marks=[]),
    pytest.param('FreemanZiemba2013public.V2-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.23113, abs=.0001), marks=[]),
])
def test_amount_gray(benchmark: str, candidate_degrees: int, image_id: str, expected: float):
    visual_degrees_test.amount_gray_test(benchmark, candidate_degrees, image_id, expected)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'FreemanZiemba2013.V1-pls',
    'FreemanZiemba2013.V2-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    ('FreemanZiemba2013public.V1-pls', 4, approx(.679954, abs=.001)),
    ('FreemanZiemba2013public.V2-pls', 4, approx(.577498, abs=.001)),
])
def test_self(benchmark, visual_degrees, expected):
    benchmark = load_benchmark(benchmark)
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score == expected
