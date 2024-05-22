import numpy as np
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_vision import load_metric


def test_perfect_score():
    metric = load_metric('value_delta')
    value_delta = metric(0.25, 0.25)
    assert value_delta == 1


def test_middle_score():
    metric = load_metric('value_delta')
    value_delta = metric(0.75, 0.25)
    assert value_delta == 0.5


def test_middle_score_reversed():
    metric = load_metric('value_delta')
    value_delta = metric(0.25, 0.75)
    assert value_delta == 0.5


def test_worst_score():
    metric = load_metric('value_delta')
    value_delta = metric(1.0, 0.0)
    assert value_delta == 0.0


def test_worst_score_reversed():
    metric = load_metric('value_delta')
    value_delta = metric(0.0, 1.0)
    assert value_delta == 0.0