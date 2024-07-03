from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.data_helpers import s3


@pytest.mark.parametrize('benchmark', [
    'BMD_2024_texture_1BehavioralAccuracyDistance',
    'BMD_2024_texture_2BehavioralAccuracyDistance',
    'BMD_2024_dotted_1BehavioralAccuracyDistance',
    'BMD_2024_dotted_2BehavioralAccuracyDistance',
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
        benchmark = f"BMD_2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling


