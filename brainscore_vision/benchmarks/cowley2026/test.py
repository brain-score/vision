import pytest

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
    def test_model_score(self):
        benchmark = load_benchmark('Cowley2026.V4-pls')
        score = benchmark(load_model('alexnet'))
        assert 0 < score.values <= 1
