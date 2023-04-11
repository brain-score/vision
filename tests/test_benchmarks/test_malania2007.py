from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.malania2007 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 5 + 2 + 2

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"Malania2007_{dataset.replace('-', '')}"
        assert identifier in benchmark_pool

    def test_mean_ceiling(self):
        benchmarks = [f"Malania2007_{dataset.replace('-', '')}" for dataset in DATASETS]
        benchmarks = [benchmark_pool[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling.sel(aggregation='center') for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.5618048355142616, abs=0.001)

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('short-2', approx(0.78719345, abs=0.001)),
        ('short-4', approx(0.49998989, abs=0.001)),
        ('short-6', approx(0.50590051, abs=0.001)),
        ('short-8', approx(0.4426336, abs=0.001)),
        ('short-16', approx(0.8383443, abs=0.001)),
        ('equal-2', approx(0.56664015, abs=0.001)),
        ('long-2', approx(0.46470421, abs=0.001)),
        ('equal-16', approx(0.44087153, abs=0.001)),
        ('long-16', approx(0.50996587, abs=0.001))
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Malania2007_{dataset.replace('-', '')}"
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center').values.item() == expected_ceiling
