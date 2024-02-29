from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmarks.scialom2024.benchmark import DATASETS
from brainscore_vision.data_helpers import s3


@pytest.mark.parametrize('dataset, expected_ceiling', [
    ('rgb', approx(0.98034, abs=0.001)),
    ('contours', approx(0.9682, abs=0.001)),
    ('phosphenes-12', approx(0.71819, abs=0.001)),
    ('phosphenes-16', approx(0.67069, abs=0.001)),
    ('phosphenes-21', approx(0.68930, abs=0.001)),
    ('phosphenes-27', approx(0.62750, abs=0.001)),
    ('phosphenes-35', approx(0.61458, abs=0.001)),
    ('phosphenes-46', approx(0.62916, abs=0.001)),
    ('phosphenes-59', approx(0.65513, abs=0.001)),
    ('phosphenes-77', approx(0.71916, abs=0.001)),
    ('phosphenes-100', approx(0.74847, abs=0.001)),
    ('segments-12', approx(0.66263, abs=0.001)),
    ('segments-16', approx(0.61250, abs=0.001)),
    ('segments-21', approx(0.63777, abs=0.001)),
    ('segments-27', approx(0.62361, abs=0.001)),
    ('segments-35', approx(0.63180, abs=0.001)),
    ('segments-46', approx(0.65888, abs=0.001)),
    ('segments-59', approx(0.71097, abs=0.001)),
    ('segments-77', approx(0.77055, abs=0.001)),
    ('segments-100', approx(0.86305, abs=0.001)),
])
def test_dataset_ceiling(self, dataset, expected_ceiling):
    benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
    benchmark = load_benchmark(benchmark)
    ceiling = benchmark.ceiling
    assert ceiling == expected_ceiling
