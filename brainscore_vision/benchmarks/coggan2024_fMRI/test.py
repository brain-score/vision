# Created by David Coggan on 2024 06 26

import pytest
from pytest import approx
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision import load_model


@pytest.mark.parametrize('region', ['V1', 'V2', 'V4', 'IT'])
def test_benchmark_registry(region):
    assert f'Coggan2024_fMRI.{region}-rdm' in benchmark_registry


@pytest.mark.parametrize('region', ['V1', 'V2', 'V4', 'IT'])
def test_benchmarks(region):
    expected_score = dict(
        V1=0.0182585,
        V2=0.3352083,
        V4=0.3008136,
        IT=0.4486508)[region]
    model = load_model('alexnet')
    benchmark = load_benchmark(f'Coggan2024_fMRI.{region}-rdm')
    score = benchmark(model)
    assert score.values == approx(expected_score)

