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
        identifier = f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"
        assert identifier in benchmark_pool

    def test_mean_ceiling(self):
        benchmarks = [f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency" for dataset in DATASETS]
        benchmarks = [benchmark_pool[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.43122, abs=0.005)

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('colour', approx(0.41543, abs=0.001)),
        ('contrast', approx(0.43703, abs=0.001)),
        ('cue-conflict', approx(0.33105, abs=0.001)),
        ('edge', approx(0.31844, abs=0.001)),
        ('eidolonI', approx(0.38634, abs=0.001)),
        ('eidolonII', approx(0.45402, abs=0.001)),
        ('eidolonIII', approx(0.45953, abs=0.001)),
        ('false-colour', approx(0.44405, abs=0.001)),
        ('high-pass', approx(0.44014, abs=0.001)),
        ('low-pass', approx(0.46888, abs=0.001)),
        ('phase-scrambling', approx(0.44667, abs=0.001)),
        ('power-equalisation', approx(0.51063, abs=0.001)),
        ('rotation', approx(0.43851, abs=0.001)),
        ('silhouette', approx(0.47571, abs=0.001)),
        ('sketch', approx(0.36962, abs=0.001)),
        ('stylized', approx(0.50058, abs=0.001)),
        ('uniform-noise', approx(0.43406, abs=0.001)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.values.item() == expected_ceiling
