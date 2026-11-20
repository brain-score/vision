import pytest
from pytest import approx
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision import load_model


@pytest.mark.parametrize('region', ['V1', 'V2', 'V3', 'V4', 'early', 'lateral', 'ventral', 'parietal'])
def test_benchmark_registry(region):
    assert f'NSD.{region}.pls' in benchmark_registry


@pytest.mark.parametrize('region', ['V1', 'V2', 'V4'])
def test_benchmarks(region):
    expected_score = dict(
        V1=0.79087484,
        V2=0.92785367,
        V4=0.88207837)[region]
    model = load_model('alexnet')
    benchmark = load_benchmark(f'NSD.{region}.pls')
    score = benchmark(model)
    assert score.values == approx(expected_score, abs=.005)

