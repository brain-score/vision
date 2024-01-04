import numpy as np
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_vision import load_metric


def test_fifty_percent():
    metric = load_metric('accuracy')
    accuracy = metric(np.array([5, 3, 6, 7, 1, 8]), np.array([5, 2, 1, 7, 1, 9]))
    assert accuracy == 0.5
