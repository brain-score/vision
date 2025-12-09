import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark, load_model


class TestExist:
    @pytest.mark.private_access
    @pytest.mark.parametrize("identifier", [
        'Papale2025.V1-ridge',
        'Papale2025.V4-ridge',
        'Papale2025.IT-ridge',
        'Papale2025.V1-ridgecv',
        'Papale2025.V4-ridgecv',
        'Papale2025.IT-ridgecv',
    ])
    def test_benchmark_registry(self, identifier):
        benchmark = load_benchmark(identifier)
        assert benchmark is not None
        assert benchmark.identifier == identifier


class TestAlexNet:
    @pytest.mark.private_access
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected_score', [
        ('Papale2025.V1-ridgecv', approx(0.7734869857127751, abs=0.001)),
        ('Papale2025.V4-ridgecv', approx(0.5851808793056532, abs=0.001)),
        ('Papale2025.IT-ridgecv', approx(0.4707093374133797, abs=0.001)),
    ])
    def test_model_score(self, benchmark, expected_score):
        benchmark = load_benchmark(benchmark)
        model = load_model('alexnet')
        score = benchmark(model)
        assert score == expected_score
