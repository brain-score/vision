import pytest

from brainscore_vision import benchmark_registry, load_model

@pytest.mark.private_access
def test_ceiling():
    benchmark = load_benchmark('Hebart2023-match')
    ceiling = benchmark.ceiling
    assert ceiling == pytest.approx(0.6767, abs=0.0001)

@pytest.mark.private_access
def test_alexnet_consistency():
    benchmark = load_benchmark('Hebart2023-match')
    benchmark.set_number_of_triplets(n=1000)
    model = load_model('alexnet')
    score = benchmark(model)
    assert score == pytest.approx(0.38, abs=0.02)
