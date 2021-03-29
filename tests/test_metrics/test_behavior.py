import os
from pathlib import Path

import pandas as pd
import xarray as xr
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.benchmarks.rajalingham2018 import load_assembly
from brainscore.metrics.image_level_behavior import I2n


@pytest.mark.private_access
class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        # assemblies
        objectome = load_assembly()
        probabilities = Path(__file__).parent / f'{model}-probabilities.nc'
        probabilities = BehavioralAssembly(xr.load_dataarray(probabilities))
        # metric
        i2n = I2n()
        score = i2n(probabilities, objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def test_ceiling(self):
        objectome = load_assembly()
        i2n = I2n()
        ceiling = i2n.ceiling(objectome)
        assert ceiling.sel(aggregation='center') == approx(.4786, abs=.0064)
        assert ceiling.sel(aggregation='error') == approx(.00537, abs=.0015)
