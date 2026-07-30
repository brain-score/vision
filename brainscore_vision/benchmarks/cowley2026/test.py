import pytest
from pytest import approx

from brainscore_vision import load_benchmark, load_model


@pytest.mark.private_access
class TestExist:
    def test_benchmark_loads(self):
        benchmark = load_benchmark('Cowley2026.V4-pls')
        assert benchmark is not None
        assert benchmark.identifier == 'Cowley2026.V4-pls'


@pytest.mark.private_access
@pytest.mark.slow
class TestAlexNet:
    @pytest.mark.parametrize('benchmark, expected_score', [
        ('Cowley2026.V4-pls', approx(0.33189280, abs=0.005)),
    ])
    def test_model_score(self, benchmark, expected_score):
        benchmark = load_benchmark(benchmark)
        score = benchmark(load_model('alexnet'))
        assert score.values == expected_score
