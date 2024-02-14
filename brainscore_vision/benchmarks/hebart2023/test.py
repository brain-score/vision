import pytest
from pytest import approx

import numpy as np
from brainio.stimuli import StimulusSet
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmarks.hebart2023 import Hebart2023Accuracy

def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry

def test_ceiling():
    benchmark = Hebart2023Accuracy()
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == approx(None, abs=None)

@pytest.mark.slow
def test_new_stimulus_set(self):
    triplets = np.array([
        self.assembly.coords["image_1"].values,
        self.assembly.coords["image_2"].values,
        self.assembly.coords["image_3"].values
    ]).T.reshape(-1, 1)

    new_ss = None
    assert len(triplets) == 453642 * 3

def test_fake_data():
    pass

def test_model_xyz_consistent():
    pass

def test_human_data():
    pass
