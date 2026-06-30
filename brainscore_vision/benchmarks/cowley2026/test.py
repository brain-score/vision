import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark, load_model

@pytest.mark.private_access
class TestExist:

    @pytest.mark.parametrize("identifier", [
        'Cowley2026_190923'
        ])
    def test_benchmark_loads(self, identifier):
        """Verify benchmark can be loaded."""
        benchmark = load_benchmark(identifier)
        assert benchmark is not None
        assert benchmark.identifier == identifier + '.V4-pls'


@pytest.mark.private_access
class TestAlexNet:
    
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected_score', [
        ('Cowley2026_190923', approx(0.61209661, abs=0.001)),
    ])
    def test_model_score(self, benchmark, expected_score):
        benchmark = load_benchmark(benchmark)
        model = load_model('alexnet')
        score = benchmark(model)
        print(score)
        assert score == expected_score