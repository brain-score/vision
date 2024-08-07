import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark


@pytest.mark.parametrize('benchmark', [
    'BMD2024.texture_1Behavioral-accuracy_distance',
    'BMD2024.texture_2Behavioral-accuracy_distance',
    'BMD2024.dotted_1Behavioral-accuracy_distance',
    'BMD2024.dotted_2Behavioral-accuracy_distance',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


class TestBehavioral:
    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('texture_1', approx(0.98283, abs=0.001)),
        ('texture_2', approx(0.97337, abs=0.001)),
        ('dotted_1', approx(0.97837, abs=0.001)),
        ('dotted_2', approx(0.93071, abs=0.001)),  # all of the above are AccuracyDistance
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"BMD2024.{dataset}Behavioral-accuracy_distance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling
