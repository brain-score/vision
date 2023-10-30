from pathlib import Path

import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import TestStandardized, TestPrecomputed, TestNumberOfTrials, \
    TestVisualDegrees
from brainscore_vision.data_helpers import s3
from .benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark

# should these be in function definitions
standardized_tests = TestStandardized()
precomputed_test = TestPrecomputed()
num_trials_test = TestNumberOfTrials()
visual_degrees_test = TestVisualDegrees()


@pytest.mark.parametrize('benchmark', [
    'movshon.FreemanZiemba2013.V1-pls',
    'movshon.FreemanZiemba2013.V2-pls',  # are these what should be checked for? what about public vs. private
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


@pytest.mark.parametrize('benchmark, expected', [
    pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.873345, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.824836, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.V1-rdm', approx(.918672, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(.856968, abs=.001),
                 marks=[pytest.mark.memory_intense]),
])
def test_ceilings(benchmark, expected):
    standardized_tests.ceilings_test(benchmark, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('movshon.FreemanZiemba2013.V1-pls', 4, approx(.668491, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.V2-pls', 4, approx(.553155, abs=.001),
                 marks=[pytest.mark.memory_intense]),
])
def test_self_regression(benchmark, visual_degrees, expected):
    standardized_tests.self_regression_test(benchmark, visual_degrees, expected)


@pytest.mark.parametrize('benchmark, visual_degrees, expected', [
    pytest.param('movshon.FreemanZiemba2013.V1-rdm', 4, approx(1, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.V2-rdm', 4, approx(1, abs=.001),
                 marks=[pytest.mark.memory_intense]),
])
def test_self_rdm(benchmark, visual_degrees, expected):
    benchmark = load_benchmark(benchmark)
    score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected
    raw_values = score.attrs['raw']
    assert hasattr(raw_values, 'split')
    assert len(raw_values['split']) == 10


@pytest.mark.memory_intense
@pytest.mark.parametrize('benchmark, expected', [
    ('movshon.FreemanZiemba2013.V1-pls', approx(.466222, abs=.005)),
    ('movshon.FreemanZiemba2013.V2-pls', approx(.459283, abs=.005)),
])
def test_FreemanZiemba2013(benchmark, expected):
    filename = 'alexnet-freemanziemba2013.aperture-private.nc'
    filepath = Path(__file__).parent / filename
    s3.download_file_if_not_exists(local_path=filepath,
                                   bucket='brainio-brainscore', remote_filepath=f'tests/test_benchmarks/{filename}')
    precomputed_test.run_test(benchmark=benchmark, precomputed_features_filepath=filename, expected=expected)


@pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
    pytest.param('movshon.FreemanZiemba2013.V1-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('movshon.FreemanZiemba2013.V1-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('movshon.FreemanZiemba2013public.V1-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.314561, abs=.0001), marks=[]),
    pytest.param('movshon.FreemanZiemba2013public.V1-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.23113, abs=.0001), marks=[]),
    pytest.param('movshon.FreemanZiemba2013.V2-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('movshon.FreemanZiemba2013.V2-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                 approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
    pytest.param('movshon.FreemanZiemba2013public.V2-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.314561, abs=.0001), marks=[]),
    pytest.param('movshon.FreemanZiemba2013public.V2-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                 approx(.23113, abs=.0001), marks=[]),
])
def test_amount_gray(benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                     brainscore_home):
    visual_degrees_test.amount_gray_test(benchmark, candidate_degrees, image_id, expected, brainio_home,
                                         resultcaching_home, brainscore_home)


@pytest.mark.private_access
@pytest.mark.parametrize('benchmark_identifier', [
    'movshon.FreemanZiemba2013.V1-pls',
    'movshon.FreemanZiemba2013.V2-pls',
])
def test_repetitions(benchmark_identifier):
    num_trials_test.repetitions_test(benchmark_identifier)


# tests for public benchmarks
@pytest.mark.parametrize('benchmark_ctr, visual_degrees, expected', [
    pytest.param(FreemanZiembaV1PublicBenchmark, 4, approx(.679954, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(FreemanZiembaV2PublicBenchmark, 4, approx(.577498, abs=.001),
                 marks=[pytest.mark.memory_intense]),
])
def test_self(benchmark_ctr, visual_degrees, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected
