import pytest

from brainscore_vision import benchmark_registry, load_model
from brainscore_vision.benchmarks.hebart2023 import Hebart2023Match

def test_benchmark_registry():
    assert "Hebart2023-match" in benchmark_registry

def test_ceiling():
    benchmark = Hebart2023Match()
    ceiling = benchmark.ceiling
    assert ceiling == pytest.approx(0.6767, abs=0.0001)

def test_alexnet_consistency():
    benchmark = Hebart2023Match()
    benchmark.set_number_of_triplets(n=1000)
    model = load_model('alexnet')
    score = benchmark(model)
    assert score == pytest.approx(0.38, abs=0.02)
