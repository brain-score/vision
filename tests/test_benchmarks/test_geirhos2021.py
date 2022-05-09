import numpy as np
import pytest
from pytest import approx

from brainscore import benchmark_pool
from brainscore.benchmarks.geirhos2021 import DATASETS


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 12 + 5

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"brendel.Geirhos2021{dataset.replace('-', '')}-cohen_kappa"
        assert identifier in benchmark_pool

    def test_mean_ceiling(self):
        benchmarks = [f"brendel.Geirhos2021{dataset.replace('-', '')}-cohen_kappa" for dataset in DATASETS]
        benchmarks = [benchmark_pool[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.431, abs=0.005)
