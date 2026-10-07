import pytest
from pytest import approx

import brainscore_vision
from brainscore_vision.benchmark_helpers import PrecomputedFeatures


def test_benchmark_runs():
    benchmark = brainscore_vision.load_benchmark('MajajHong2015.IT-spatial_correlation')
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    features = PrecomputedFeatures(source, visual_degrees=8)
    benchmark(features)
