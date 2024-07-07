import numpy as np
import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark, load_model
from brainscore_vision.benchmarks.malania2007.benchmark import DATASETS


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 5 + 2 + 2 + 1

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Malania2007.{dataset}"
        assert identifier in benchmark_registry

    @pytest.mark.private_access
    def test_mean_ceiling(self):
        benchmarks = [f"Malania2007.{dataset}" for dataset in DATASETS]
        benchmarks = [load_benchmark(benchmark) for benchmark in benchmarks]
        ceilings = [benchmark.ceiling for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.5757928329186803, abs=0.001)

    # these test values are for the pooled score ceiling
    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('short2-threshold_elevation', approx(0.78719345, abs=0.001)),
        ('short4-threshold_elevation', approx(0.49998989, abs=0.001)),
        ('short6-threshold_elevation', approx(0.50590051, abs=0.001)),
        ('short8-threshold_elevation', approx(0.4426336, abs=0.001)),
        ('short16-threshold_elevation', approx(0.8383443, abs=0.001)),
        ('equal2-threshold_elevation', approx(0.56664015, abs=0.001)),
        ('long2-threshold_elevation', approx(0.46470421, abs=0.001)),
        ('equal16-threshold_elevation', approx(0.44087153, abs=0.001)),
        ('long16-threshold_elevation', approx(0.50996587, abs=0.001)),
        ('vernieracuity-threshold', approx(0.70168481, abs=0.001))
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Malania2007.{dataset}"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_score', [
        ('short2-threshold_elevation', approx(0.0, abs=0.001)),
        ('short4-threshold_elevation', approx(0.0, abs=0.001)),
        ('short6-threshold_elevation', approx(0.0, abs=0.001)),
        ('short8-threshold_elevation', approx(0.0, abs=0.001)),
        ('short16-threshold_elevation', approx(0.0, abs=0.001)),
        ('equal2-threshold_elevation', approx(0.0, abs=0.001)),
        ('long2-threshold_elevation', approx(0.0, abs=0.001)),
        ('equal16-threshold_elevation', approx(0.0, abs=0.001)),
        ('long16-threshold_elevation', approx(0.0, abs=0.001)),
        ('vernieracuity-threshold', approx(0.0, abs=0.001))
    ])
    def test_model_score(self, dataset, expected_score):
        benchmark = f"Malania2007.{dataset}"
        benchmark = load_benchmark(benchmark)
        model = load_model('alexnet')
        model_score = benchmark(model)
        assert model_score.values == expected_score
