import pytest

from brainscore_vision import benchmark_registry, load_model
from brainscore_vision.benchmarks.hebart2023 import Hebart2023Accuracy

def test_benchmark_registry():
    assert "Hebart2023-acc" in benchmark_registry

def test_ceiling():
    benchmark = Hebart2023Accuracy()
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == pytest.approx(0.6844, abs=0.0001)

def test_alexnet_consistency():
    benchmark = Hebart2023Accuracy()
    benchmark.set_number_of_triplets(n=1000)
    model = load_model('alexnet')
    score = benchmark(model)
    assert score.sel(aggregation='center') == pytest.approx(0.38, abs=0.02)