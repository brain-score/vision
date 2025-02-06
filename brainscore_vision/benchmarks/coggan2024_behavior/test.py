# Created by David Coggan on 2024 06 26

import pytest
from pytest import approx
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision import load_model


def test_benchmark_registry():
    assert ('tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity' in
            benchmark_registry)

@pytest.mark.private_access
def test_benchmarks():
    benchmark = load_benchmark(
        'tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity')
    model = load_model('alexnet')
    result = benchmark(model)
    assert result.values == approx(0.1318, abs=.001)


